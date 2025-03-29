import gc
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sys
from memory_profiler import profile

from mel_dataset import AudioDataset
from config import get_config, get_tok_weights_file_path
from model.mel.mel_vqvae import VQVAE
from utils import calculate_accuracy, get_loaders


def validate(model, dataloader, epoch, writer, device):
    model.eval()
    model.quantizer.training = False
    val_loss = 0
    val_acc = 0
    model = model.to(device)
    batch_iterator = tqdm(dataloader, desc=f'Validating epoch {epoch + 1:02d}')
    with (torch.no_grad()):
        for batch in batch_iterator:
            original = batch["features"]
            inputs = batch["augment"]
            inputs = inputs.to(device)
            quant_loss, recon, info = model(inputs)

            loss = quant_loss + reconstruction_loss(recon, original)
            val_loss += loss.item()
            accuracy = calculate_accuracy(recon, original)
            val_acc += accuracy
    avg_val_loss = val_loss / len(val_loader)
    avg_val_accuracy = val_acc / len(val_loader)
    print(f'Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}')

    writer.add_scalar('val_loss', avg_val_loss, epoch + 1)
    writer.add_scalar('val_accuracy', avg_val_accuracy, epoch + 1)
    writer.flush()


def train(model, train_loader, val_loader, config, appendix):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    writer = SummaryWriter(config["training_params"][
                               "experiment_name_tok"] + "/" + appendix)

    initial_epoch = 0
    global_step = 0
    """
    if config["training_params"]["preload"]:
        model_filename = get_tok_weights_file_path(config,
                                               config["training_params"][
                                                   "preload"],
                                               appendix)
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    """

    optimizer = optim.Adam(model.parameters(),
                           lr=config['training_params']['lr'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model = model.to(device)

    for epoch in range(initial_epoch, config['training_params']['epochs_tok']):
        model.train()
        model.quantizer.training = True
        total_loss = 0
        total_accuracy = 0
        batch_iterator = tqdm(train_loader, desc=f'Processing epoch {epoch + 1:02d}')
        for batch in batch_iterator:
            original = batch["features"]
            inputs = batch["augmented"]
            inputs = inputs.to(device)
            optimizer.zero_grad()
            quant_loss, recon, info = model(inputs)
            loss = quant_loss + reconstruction_loss(recon.cpu(), original)

            total_loss += loss.item()
            accuracy = calculate_accuracy(recon.cpu(), original)
            total_accuracy += accuracy
            #codes_replaced = model.quantizer.num_codes_replaced
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}",
                                        f"accuracy": f"{accuracy:6.3f}",
                                        f"perplexity": f"{info[0]:6.3f}"})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('train_accuracy', accuracy, global_step)
            #writer.add_scalar('Num of codes replaced', codes_replaced, global_step)
            writer.flush()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            gc.collect()
            torch.cuda.empty_cache()

            global_step += 1

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)
        print(f'Epoch {epoch + 1:02d} - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}')

        # Do validation
        if (epoch + 1) % 5 == 0:
            validate(model, val_loader, epoch, writer, device)
        scheduler.step()

        # Save model at the end of every epoch
        model_filename = get_tok_weights_file_path(config, f'{epoch + 1:02d}', appendix)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def reconstruction_loss(reconstructed, original):
    MSE = F.mse_loss(reconstructed, original, reduction='mean')
    return MSE


if __name__ == "__main__":
    orig_path = "../dataset/orig"
    chip_path = "../dataset/chip"

    assert sys.argv[1].lower() == "orig" or sys.argv[1].lower() == "chip",\
        "Invalid filepath included; Use either Orig or Chip"

    config = get_config()

    dataset = None
    appendix = ""

    for i in range(2):
        model = VQVAE(config=config)
        if i == 0:
            appendix = "orig"
            dataset = AudioDataset(orig_path,
                                   seq_len=config["model_params"]["seq_len"],
                                   sample_rate=config["model_params"]["sample_rate"],
                                   path_to_info_file="../dataset/dataset_info_orig.txt")
        else:
            appendix = "chip"
            dataset = AudioDataset(chip_path,
                                   seq_len=config["model_params"]["seq_len"],
                                   sample_rate=config["model_params"]["sample_rate"],
                                   path_to_info_file="../dataset/dataset_info_chip.txt")

        train_loader, val_loader = get_loaders(dataset,
                                               config["training_params"][
                                                   "shuffle_dataset"],
                                               config["training_params"][
                                                   "validation_split"],
                                               config["training_params"][
                                                   "batch_size"])
        train(model, train_loader, val_loader, config, appendix)
