import warnings

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

import sys

from dataset_wave import AudioDataset
from config import get_config, get_weights_file_path
from model.vqvae import get_model


def calculate_accuracy(pred, tgt, threshold=0.1):
    correct_predictions = (torch.abs(pred - tgt) < threshold).sum().item()
    accuracy = correct_predictions / tgt.numel()
    return accuracy


def get_loaders(dataset, shuffle_dataset, validation_split, batch_size):
    random_seed = 42
    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size} samples")
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


def validate(model, dataloader, epoch, writer):
    model.eval()
    model.quantizer.training = False
    val_loss = 0
    val_acc = 0
    batch_iterator = tqdm(dataloader, desc=f'Validating epoch {epoch + 1:02d}')
    with (torch.no_grad()):
        for batch in batch_iterator:
            with autocast(device_type="cuda"):
                outputs = model(batch)
                codebook_loss = outputs["quantized_losses"]["codebook_loss"]
                commitment_loss = outputs["quantized_losses"][
                    "commitment_loss"]
                diversity_loss = outputs["quantized_losses"]["diversity_loss"]
                recon_loss = reconstruction_loss(outputs["generated_image"],
                                                 batch)

                loss = codebook_loss + commitment_loss + diversity_loss + recon_loss
            val_loss += loss.item()
            accuracy = calculate_accuracy(outputs["generated_image"], batch)
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
                               "experiment_name"] + "/" + appendix)

    optimizer = optim.Adagrad(model.parameters(),
                           lr=config['training_params']['lr'])
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = GradScaler()

    initial_epoch = 0
    global_step = 0
    if config["training_params"]["preload"]:
        model_filename = get_weights_file_path(config,
                                               config["training_params"][
                                                   "preload"],
                                               appendix)
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    for epoch in range(initial_epoch, config['training_params']['epochs']):
        model.train()
        model.quantizer.training = True
        total_loss = 0
        total_accuracy = 0
        batch_iterator = tqdm(train_loader, desc=f'Processing epoch {epoch + 1:02d}')
        for batch in batch_iterator:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            with (autocast(device_type=device.type)):
                outputs = model(batch)

                codebook_loss = outputs["quantized_losses"]["codebook_loss"]
                commitment_loss = outputs["quantized_losses"]["commitment_loss"]
                diversity_loss = outputs["quantized_losses"]["diversity_loss"]
                recon_loss = reconstruction_loss(outputs["generated_image"], batch)
                perplexity = outputs["quantized_losses"]["perplexity"]

                loss = codebook_loss + commitment_loss + diversity_loss + recon_loss

            total_loss += loss.item()
            accuracy = calculate_accuracy(outputs["generated_image"], batch)
            total_accuracy += accuracy
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}",
                                        f"accuracy": f"{accuracy:6.3f}",
                                        f"perplexity": f"{perplexity:6.3f}"})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('codebook_loss', codebook_loss, global_step)
            writer.add_scalar('commitment_loss', commitment_loss, global_step)
            writer.add_scalar('diversity_loss', diversity_loss, global_step)
            writer.add_scalar('reconstruction_loss', recon_loss, global_step)
            writer.add_scalar('train_accuracy', accuracy, global_step)
            writer.add_scalar('perplexity', perplexity, global_step)
            writer.flush()

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

        avg_train_loss = total_loss / len(train_loader)
        avg_train_accuracy = total_accuracy / len(train_loader)
        print(f'Epoch {epoch + 1:02d} - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}')

        # Do validation
        if (epoch + 1) % 5 == 0:
            validate(model, val_loader, epoch, writer)
        #scheduler.step()

        # Save model at the end of every epoch
        model_filename = get_weights_file_path(config, f'{epoch + 1:02d}', appendix)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

def reconstruction_loss(reconstructed, original):
    return nn.MSELoss()(reconstructed, original)


if __name__ == "__main__":
    orig_path = "../dataset/orig"
    chip_path = "../dataset/chip"

    warnings.filterwarnings('ignore')

    assert sys.argv[1].lower() == "orig" or sys.argv[1].lower() == "chip",\
        "Invalid filepath included; Use either Orig or Chip"

    config = get_config()
    model = get_model(config)

    dataset = None
    appendix = ""

    for i in range(2):
        if i == 0:
            appendix = "orig"
            dataset = AudioDataset(orig_path,
                                   config["model_params"]["seq_len"],
                                   config["model_params"]["sample_rate"])
        else:
            appendix = "chip"
            dataset = AudioDataset(chip_path,
                                   config["model_params"]["seq_len"],
                                   config["model_params"]["sample_rate"])

        train_loader, val_loader = get_loaders(dataset,
                                               config["training_params"][
                                                   "shuffle_dataset"],
                                               config["training_params"][
                                                   "validation_split"],
                                               config["training_params"][
                                                   "batch_size"])
        train(model, train_loader, val_loader, config, appendix)
    """
    if sys.argv[1].lower() == "orig":
        dataset = AudioDataset(orig_path, config["model_params"]["seq_len"], config["model_params"]["sample_rate"])
        appendix = "orig"
    elif sys.argv[1].lower() == "chip":
        dataset = AudioDataset(chip_path, config["model_params"]["seq_len"], config["model_params"]["sample_rate"])
        appendix = "chip"
    

    train_loader, val_loader = get_loaders(dataset,
                                           config["training_params"][
                                               "shuffle_dataset"],
                                           config["training_params"][
                                               "validation_split"],
                                           config["training_params"][
                                               "batch_size"])
    train(model, train_loader, val_loader, config, appendix)
    """
