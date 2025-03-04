import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import librosa
from torchvision.utils import make_grid
from dataset_wave import AudioDataset
from model.vqvae import get_model
from config import get_config, get_weights_file_path

def load_model(config, epoch, appendix):
    model = get_model(config)
    model_filename = get_weights_file_path(config, epoch, appendix)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model

def show_images(original_images, reconstructed_images, title):
    fig, ax = plt.subplots(8, 4, sharex=True, figsize=(15, 15))
    for i in range(8):
        for j in range(2):
            # Original images
            ax[i][j].set_title(f'{title}_Original_{i}_{j}')
            original_channel = original_images[i][j].cpu().numpy()
            spec_orig = librosa.display.specshow(original_channel, x_axis="time", sr=16000, ax=ax[i][j])
            fig.colorbar(spec_orig, ax=ax[i][j], format="%+2.0f dB")
            ax[i][j].axis('off')

            # Reconstructed images
            ax[i][j+2].set_title(f'{title}_Reconstructed_{i}_{j}')
            reconstructed_channel = reconstructed_images[i][j].cpu().numpy()
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
    original_images = []
    reconstructed_images = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch)
            reconstructed = outputs["generated_image"]
            original_images.append(batch)
            reconstructed_images.append(reconstructed)
            if len(original_images) >= 8:
                break

    original_images = torch.cat(original_images)[:8]
    reconstructed_images = torch.cat(reconstructed_images)[:8]

    show_images(original_images, reconstructed_images, 'Images')

if __name__ == "__main__":
    config = get_config()
    appendix = "orig"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = AudioDataset("../dataset/orig", config["model_params"]["seq_len"], config["model_params"]["sample_rate"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = load_model(config,10 ,appendix)
    model.to(device)

    evaluate_model(model, dataloader, device)
    visualize_codebook(model.quantizer)
