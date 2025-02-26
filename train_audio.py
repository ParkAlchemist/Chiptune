import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler

from dataset_audio import AudioDataset
from model_audio import build_transformer

from config_audio import get_weights_file_path, get_config

from torch.utils.tensorboard import SummaryWriter

import warnings
from pathlib import Path
from tqdm import tqdm


orig_path = "dataset\orig"
chip_path = "dataset\chip"

time_masking = T.TimeMasking(time_mask_param=30)
freq_masking = T.FrequencyMasking(freq_mask_param=15)


def calculate_accuracy(pred, tgt, threshold=0.01):
    correct_predictions = (torch.abs(pred - tgt) < threshold).sum().item()
    accuracy = correct_predictions / tgt.numel()
    return accuracy


def run_validation(model, val_loader, seg_len, channels, n_mels, device, loss_fn, epoch, augment):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        batch_iterator = tqdm(val_loader, desc=f"Validating Epoch {epoch:02d}")
        for batch in batch_iterator:
            torch.cuda.empty_cache()

            original_batch = batch["original"]\
                .view(-1, channels, n_mels, seg_len).to(device)
            chiptune_batch = batch["chiptune"]\
                .view(-1, channels, n_mels, seg_len).to(device)

            encoder_input = original_batch
            decoder_input = chiptune_batch

            if augment:
                encoder_input = freq_masking(time_masking(encoder_input))
                decoder_input = freq_masking(time_masking(decoder_input))

            encoder_mask = None
            decoder_mask = None

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask,
                                          decoder_input, decoder_mask)

            label = chiptune_batch

            loss = loss_fn(decoder_output, label)
            val_loss += loss.item()
            accuracy = calculate_accuracy(decoder_output, label)
            val_acc += accuracy
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}",
                                        f"accuracy": f"{accuracy:6.3f}"})

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f} - Validation Accuracy: {avg_val_acc:.4f}')
    return avg_val_loss, avg_val_acc


def get_loaders(dataset, shuffle_dataset, validation_split, batch_size):

    random_seed = 42
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} pairs of samples")
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler, num_workers=4)

    return train_loader, val_loader


def train_model(config_audio):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device {device}')

    Path(config_audio["model_folder"]).mkdir(parents=True, exist_ok=True)

    dataset = AudioDataset(orig_path, chip_path,
                           segment_length=config_audio["seg_len"],
                           sample_rate=config_audio["sr"],
                           n_mels=config_audio["n_mels"])

    validation_split = .2

    train_loader, val_loader = get_loaders(dataset, False, validation_split,
                                           config_audio["batch_size"])

    model = build_transformer((config_audio["channels"], config_audio["n_mels"], config_audio["seg_len"]),
                              (config_audio["channels"], config_audio["n_mels"], config_audio["seg_len"]),
                              config_audio["seg_len"],
                              config_audio["seg_len"],
                              config_audio["batch_size"],
                              config_audio["d_model"]).to(device)
    # Tensorboard
    writer = SummaryWriter(config_audio["experiment_name"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config_audio["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config_audio["preload"]:
        model_filename = get_weights_file_path(config_audio,
                                               config_audio["preload"])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    loss_fn = nn.MSELoss().to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        total_loss = 0
        total_accuracy = 0
        batch_iterator = tqdm(train_loader, desc=f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:
            torch.cuda.empty_cache()

            original_batch = batch["original"]\
                .view(-1, config_audio["channels"], config_audio["n_mels"], config_audio["seg_len"]).to(device)
            chiptune_batch = batch["chiptune"]\
                .view(-1, config_audio["channels"], config_audio["n_mels"], config_audio["seg_len"]).to(device)

            encoder_input = original_batch
            decoder_input = chiptune_batch

            if config_audio["augment"]:
                encoder_input = freq_masking(time_masking(encoder_input))
                decoder_input = freq_masking(time_masking(decoder_input))

            encoder_mask = None
            decoder_mask = None

            # Run the tensors through the transformers
            encoder_output = model.encode(encoder_input, encoder_mask)  # (b, seq_len, d_model)
            decoder_output = model.decode(encoder_output,
                                          encoder_mask, decoder_input,
                                          decoder_mask)  # (b, seq_len, d_model)

            label = chiptune_batch  # (b, seq_len)

            # (b, seq_len, tgt_vocab) --> (b * seq_len, tgt_vocab_size)
            loss = loss_fn(decoder_output, label)
            total_loss += loss.item()
            accuracy = calculate_accuracy(decoder_output, label)
            total_accuracy += accuracy
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}", f"accuracy": f"{accuracy:6.3f}"})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('train_accuracy', accuracy, global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)
        print(f'Epoch {epoch+1:02d} - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}')

        avg_val_loss, avg_val_acc = run_validation(model, val_loader,
                                      config_audio["seg_len"], config_audio["channels"], config_audio["n_mels"], device, loss_fn, epoch, config_audio["augment"])
        writer.add_scalar('val_loss', avg_val_loss, global_step)
        writer.add_scalar('val_acc', avg_val_acc, global_step)
        writer.flush()

        # Save model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
