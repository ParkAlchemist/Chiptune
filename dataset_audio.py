import math

import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Any
import os


class AudioDataset(Dataset):
    def __init__(self, original_dir, chiptune_dir, sample_rate=16000,
                 segment_length=128, n_mels=80) -> None:
        super().__init__()
        self.original_file_paths = self.get_all_files(original_dir)
        self.chiptune_file_paths = self.get_all_files(chiptune_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_mels = n_mels

        assert len(self.original_file_paths) == len(
            self.chiptune_file_paths), "Mismatch in number of files between directories"

        self.segment_info = []
        self.total_segments = 0

        for original_file, chiptune_file in zip(self.original_file_paths,
                                                self.chiptune_file_paths):
            original_waveform = self.load_audio(original_file)
            #chiptune_waveform = self.load_audio(chiptune_file)

            original_spectrogram = self.audio_to_spectrogram(original_waveform)
            #chiptune_spectrogram = self.audio_to_spectrogram(chiptune_waveform)

            original_spectrogram = self.normalize_spectrogram(
                original_spectrogram)
            #chiptune_spectrogram = self.normalize_spectrogram(
            #    chiptune_spectrogram)

            num_segments = math.ceil(original_spectrogram.size(2) / self.segment_length)
            self.segment_info.append(
                (original_file, chiptune_file, num_segments))
            self.total_segments += num_segments

    def __len__(self):
        return self.total_segments

    def __getitem__(self, index: Any) -> Any:

        track_idx = 0

        for i in range(len(self.segment_info)):
            track_idx = i
            if index >= self.segment_info[track_idx][2]:
                index = index - self.segment_info[track_idx][2]
                continue
            else: break

        segment_idx = index
        original_file, chiptune_file, num_segments = self.segment_info[track_idx]

        original_waveform = self.load_audio(original_file)
        chiptune_waveform = self.load_audio(chiptune_file)

        original_spectrogram = self.audio_to_spectrogram(
            original_waveform)
        chiptune_spectrogram = self.audio_to_spectrogram(
            chiptune_waveform)

        original_spectrogram = self.normalize_spectrogram(
            original_spectrogram)
        chiptune_spectrogram = self.normalize_spectrogram(
            chiptune_spectrogram)

        if segment_idx == num_segments - 1:
            pad_length = self.segment_length - original_spectrogram.size(2) % self.segment_length
            padding = torch.zeros(original_spectrogram.size(0), original_spectrogram.size(1), pad_length)
            original_spectrogram = torch.cat((original_spectrogram, padding), dim=1)
            chiptune_spectrogram = torch.cat((chiptune_spectrogram, padding), dim=1)

        start_idx = segment_idx * self.segment_length
        end_idx = start_idx + self.segment_length

        original_segment = original_spectrogram[:, :, start_idx:end_idx]
        chiptune_segment = chiptune_spectrogram[:, :, start_idx:end_idx]

        return {
            "original": original_segment,
            "chiptune": chiptune_segment
        }

    def get_all_files(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if
                os.path.isfile(os.path.join(directory, f))]

    def load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(
                waveform)
        return waveform

    def audio_to_spectrogram(self, waveform):
        spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=self.n_mels
        )(waveform)
        spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
        return spectrogram

    def normalize_spectrogram(self, spectrogram):
        mean = spectrogram.mean()
        std = spectrogram.std()
        return (spectrogram - mean) / std

    def segment_spectrogram(self, spectrogram):
        segments = []
        for i in range(0, spectrogram.size(2) - self.segment_length + 1,
                       self.segment_length):
            segments.append(spectrogram[:, :, i:i + self.segment_length])
        return torch.stack(segments)
