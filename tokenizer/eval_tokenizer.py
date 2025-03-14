import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from dataset_wave import AudioDataset
from model.vqvae import get_model
from config import get_config, get_tok_weights_file_path

def load_model(config, epoch, appendix):
    model = get_model(config)
    model_filename = get_tok_weights_file_path(config, f"{epoch:02d}", appendix)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model

def show_images(original_images, reconstructed_images, title):
    fig, ax = plt.subplots(4, 4, sharex=True, figsize=(15, 15))
    for i in range(4):
        for j in range(2):
            # Original images
            ax[i][j].set_title(f'{title}_Original_{i}_{j}')
            original_channel = original_images[i][j]
            spec_orig = librosa.display.specshow(original_channel, x_axis="time", sr=16000, ax=ax[i][j])
            fig.colorbar(spec_orig, ax=ax[i][j], format="%+2.0f dB")
            ax[i][j].axis('off')

            # Reconstructed images
            ax[i][j+2].set_title(f'{title}_Reconstructed_{i}_{j}')
            reconstructed_channel = reconstructed_images[i][j]
            spec_recon = librosa.display.specshow(reconstructed_channel, x_axis="time", sr=16000, ax=ax[i][j+2])
            fig.colorbar(spec_recon, ax=ax[i][j+2], format="%+2.0f dB")
            ax[i][j+2].axis('off')
    plt.tight_layout()
    plt.show()


def visualize_codebook(quantizer):
    codebook = quantizer.embedding.weight.detach().cpu().numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(codebook, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Codebook Visualization')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Codebook Entries')
    plt.show()

def evaluate_model(model, dataloader, device):
    model.eval()
    model.quantizer.training = False
    original_images = []
    reconstructed_images = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            reconstructed = outputs["generated_image"]
            original_images.append(batch.cpu().numpy())
            reconstructed_images.append(reconstructed.cpu().numpy())
            if len(original_images) >= 4:
                break

    original_images = np.concatenate(original_images)[:4]
    reconstructed_images = np.concatenate(reconstructed_images)[:4]

    show_images(original_images, reconstructed_images, 'Images')
    """
    accuracy = accuracy_score(original_images, reconstructed_images)
    precision = precision_score(original_images, reconstructed_images, average='weighted')
    recall = recall_score(original_images, reconstructed_images, average='weighted')
    f1 = f1_score(original_images, reconstructed_images, average='weighted')
    conf_matrix = confusion_matrix(original_images, reconstructed_images)
    roc_auc = roc_auc_score(original_images, reconstructed_images, multi_class='ovr')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    """


if __name__ == "__main__":
    config = get_config()
    appendix = "orig"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = AudioDataset("../dataset/orig", config["model_params"]["seq_len"], config["model_params"]["sample_rate"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = load_model(config,6 ,appendix)
    model.to(device)

    evaluate_model(model, dataloader, device)
    visualize_codebook(model.quantizer)
