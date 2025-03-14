import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

import sys

from dataset_wave import AudioDataset
from config import get_config, get_enc_weights_file_path
from model.encoder import get_encoder
from model.decoder import get_decoder
from utils import calculate_accuracy, get_loaders


def train_encoder(encoder, decoder, train_loader, val_loader, config, appendix):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    writer = SummaryWriter(config["training_params"]["experiment_name_enc"] + "/" + appendix)

    optimizer = optim.Adam(encoder.parameters(), lr=config['training_params']['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler()

    initial_epoch = 0
    global_step = 0
    if config["training_params"]["load_encoder"]:
        encoder_filename = get_enc_weights_file_path(config,
                                                 f"{config['training_params']['load_encoder']:02d}",
                                                 appendix)
        print(f'Preloading encoder {encoder_filename}')
        state = torch.load(encoder_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    for epoch in range(initial_epoch, config['training_params']['epochs_enc']):
        encoder.train()
        decoder.eval()  # Ensure decoder is not updated
        total_loss = 0
        total_accuracy = 0
        batch_iterator = tqdm(train_loader, desc=f'Processing epoch {epoch + 1:02d}')
        for batch in batch_iterator:
            batch = batch.to(device)
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                z_e = encoder(batch)
                recon_x = decoder(z_e)

                recon_loss = reconstruction_loss(recon_x, batch)
                loss = recon_loss

            total_loss += loss.item()
            accuracy = calculate_accuracy(recon_x, batch)
            total_accuracy += accuracy
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}", f"accuracy": f"{accuracy:6.3f}"})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('reconstruction_loss', recon_loss, global_step)
            writer.add_scalar('train_accuracy', accuracy, global_step)
            writer.flush()

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)
        print(f'Epoch {epoch + 1:02d} - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}')

        # Do validation
        if (epoch + 1) % 1 == 0:
            validate(encoder, decoder, val_loader, epoch, writer, device)
        scheduler.step()

        # Save model at the end of every epoch
        model_filename = get_enc_weights_file_path(config, f'{epoch + 1:02d}', appendix)
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def reconstruction_loss(reconstructed, original):
    MSE = F.mse_loss(reconstructed, original, reduction='sum')
    return MSE

def validate(encoder, decoder, val_loader, epoch, writer, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            z_e = encoder(batch)
            recon_x = decoder(z_e)
            loss = reconstruction_loss(recon_x, batch)
            total_loss += loss.item()
            accuracy = calculate_accuracy(recon_x, batch)
            total_accuracy += accuracy

    avg_val_loss = total_loss / len(val_loader)
    avg_val_accuracy = total_accuracy / len(val_loader)
    writer.add_scalar('val_loss', avg_val_loss, epoch + 1)
    writer.add_scalar('val_accuracy', avg_val_accuracy, epoch + 1)
    print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}')

# Example usage
if __name__ == "__main__":
    orig_path = "../dataset/orig"
    chip_path = "../dataset/chip"

    warnings.filterwarnings('ignore')

    config = get_config()
    decoder = get_decoder(config)

    dataset = None
    appendix = ""

    for i in range(2):
        encoder = get_encoder(config)
        if i == 0:
            appendix = "orig"
            dataset = AudioDataset(orig_path, config["model_params"]["seq_len"], config["model_params"]["sample_rate"], path_to_info_file="../dataset/dataset_info_orig.txt")
        else:
            appendix = "chip"
            dataset = AudioDataset(chip_path, config["model_params"]["seq_len"], config["model_params"]["sample_rate"], path_to_info_file="../dataset/dataset_info_chip.txt")

        train_loader, val_loader = get_loaders(dataset, config["training_params"]["shuffle_dataset"], config["training_params"]["validation_split"], config["training_params"]["batch_size"])
        train_encoder(encoder, decoder, train_loader, val_loader, config, appendix)
