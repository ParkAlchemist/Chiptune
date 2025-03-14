import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from enum import Enum


class Feature(Enum):
    MFCC = 0
    MFCC_DELTA = 1
    MFCC_DELTA_DELTA = 2
    MEL_SPECTROGRAM = 3
    CQT = 4
    CHROMA = 5
    CHROMA_CQT = 6
    SPECTRAL_CENTROID = 7
    SPECTRAL_CONTRAST = 8
    SPECTRAL_FLATNESS = 9
    SPECTRAL_ROLLOFF = 10
    ONSET = 11
    TEMPO = 12
    HPSS_HARMONIC = 13
    HPSS_PERCUSSIVE = 14
    RMS = 15
    ZCR = 16


def calculate_accuracy(pred, tgt, threshold=0.001):
    correct_predictions = (torch.abs(pred - tgt) < threshold).sum().item()
    accuracy = correct_predictions / tgt.numel()
    return accuracy


def get_loaders(dataset, shuffle_dataset: bool, validation_split: float, batch_size: int):
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
                            sampler=val_sampler, num_workers=2)

    return train_loader, val_loader
