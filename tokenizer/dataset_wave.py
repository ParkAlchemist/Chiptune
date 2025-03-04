import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Any
import os


from feature_extractor import FeatureExtractor


class AudioDataset(Dataset):
    def __init__(self, dir, seq_len, sample_rate=16000, overlap_percentage=0.5, n_mfcc=13) -> None:
        super().__init__()
        self.file_paths = self.get_all_files(dir)
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.orig_sr = 48000
        self.overlap_percentage = overlap_percentage
        self.segment_info = []
        self.total_segments = 0
        self.feature_extractor = FeatureExtractor(sample_rate, n_mfcc)

        for file in self.file_paths:
            waveform = self.load_audio(file)
            snippets = self.split_audio_with_overlap(waveform)
            for i in range(len(snippets)):
                self.segment_info.append((file, i))
                self.total_segments += 1

    def __len__(self):
        return self.total_segments

    def __getitem__(self, index: Any) -> Any:
        # Fetch audio snippet
        file, snippet_idx = self.segment_info[index]
        snippet_length = int(self.seq_len * self.orig_sr)
        overlap_length = int(snippet_length * self.overlap_percentage)
        step = snippet_length - overlap_length
        offset = snippet_idx * step

        snippet = self.load_audio(file, offset, snippet_length)

        # Ensure the snippet is the correct length
        if snippet.size(1) < self.seq_len * self.sample_rate:
            pad_length = self.seq_len - snippet.size(1)
            padding = torch.zeros(snippet.size(0), pad_length)
            snippet = torch.cat((snippet, padding), dim=1)
        elif snippet.size(1) > self.seq_len * self.sample_rate:
            snippet = snippet[:, :self.seq_len]

        # Extract features
        features = self.feature_extractor.extract_features(snippet.detach().cpu().numpy())

        return features

    @staticmethod
    def get_all_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def load_audio(self, file_path, offset=0, n_frames=-1):
        waveform, sr = torchaudio.load(file_path, frame_offset=offset, num_frames=n_frames)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform

    def split_audio_with_overlap(self, waveform):
        snippet_length = int(self.sample_rate * self.seq_len)
        overlap_length = int(snippet_length * self.overlap_percentage)
        step = snippet_length - overlap_length
        snippets = [waveform[:, i:i + snippet_length] for i in range(0, waveform.size(1) - snippet_length + 1, step)]
        return snippets
