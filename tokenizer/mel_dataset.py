import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Any
import os
from memory_profiler import profile

import timeit

from mel_feature_extractor import FeatureExtractor


class AudioDataset(Dataset):
    def __init__(self, dir, seq_len, sample_rate=22050, overlap_percentage=0.5, path_to_info_file=None) -> None:
        super().__init__()
        self.file_paths = np.array(self.get_all_files(dir))
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.overlap_percentage = overlap_percentage

        self.segment_info = np.empty([1, 3])
        self.total_segments = 0
        self.n_augment_methods = 3

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=50)
        self.frequency_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=50)

        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)

        if path_to_info_file:
            with open(f"{path_to_info_file}", "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    file_name, idx, sr = line.split("|")
                    self.segment_info = np.vstack((self.segment_info, np.array([(file_name, int(idx), int(sr))])))
                self.segment_info = np.delete(self.segment_info, 0, axis=0)
                self.total_segments = len(self.segment_info)
        else:
            for file in self.file_paths:
                waveform, orig_sr = self.load_audio(file)
                snippets = self.split_audio_with_overlap(waveform)
                for i in range(len(snippets)):
                    self.segment_info = np.vstack((self.segment_info, np.array((file, i, orig_sr))))
                self.segment_info = np.delete(self.segment_info, 0, axis=0)
                self.total_segments = len(self.segment_info)

    def __len__(self):
        return self.total_segments * (self.n_augment_methods + 1)

    def __getitem__(self, index: Any) -> Any:
        aug_idx = index // self.total_segments
        seg_idx = index % self.total_segments
        # Fetch audio snippet
        file, snippet_idx, sr = self.segment_info[seg_idx]
        snippet_length = int(self.seq_len * int(sr))
        overlap_length = int(snippet_length * self.overlap_percentage)
        step = snippet_length - overlap_length
        offset = int(snippet_idx) * step

        snippet, orig_sr = self.load_audio(file, offset, snippet_length)

        # Ensure the snippet is the correct length
        if snippet.shape[1] < self.seq_len * self.sample_rate:
            pad_length = (self.seq_len * self.sample_rate) - snippet.shape[1]
            padding = torch.zeros(snippet.shape[0], pad_length)
            snippet = torch.cat((snippet, padding), dim=1)
        elif snippet.shape[1] > self.seq_len * self.sample_rate:
            snippet = snippet[:, :self.seq_len * self.sample_rate]

        # Extract features
        features = self.feature_extractor.get_mel_spec(snippet.detach().numpy())

        # Augment data
        augmented_features = self.augment_data(torch.from_numpy(features), aug_idx)

        return {"features": features, "augmented": augmented_features}

    @staticmethod
    def get_all_files(directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def load_audio(self, file_path, offset=0, n_frames=-1):
        with open(file_path, 'rb') as f:
            waveform, sr = torchaudio.load(f, frame_offset=offset, num_frames=n_frames)
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform, sr

    def split_audio_with_overlap(self, waveform):
        snippet_length = int(self.sample_rate * self.seq_len)
        overlap_length = int(snippet_length * self.overlap_percentage)
        step = snippet_length - overlap_length
        snippets = [waveform[:, i:i + snippet_length] for i in range(0, waveform.size(1) - snippet_length + 1, step)]
        return snippets

    def augment_data(self, spec, aug_idx):

        if aug_idx == 1:
            # Time masking
            spec = self.time_mask(spec)
            return spec

        elif aug_idx == 2:
            # Frequency masking
            spec = self.frequency_mask(spec)
            return spec

        elif aug_idx == 3:
            # Time and Frequency masking
            spec = self.time_mask(spec)
            spec = self.frequency_mask(spec)
            return spec

        return spec



if __name__ == "__main__":
    dir = "../dataset"
    appendix = "chip"
    info_file_path = f"{dir}/dataset_info_{appendix}.txt"

    dataset = AudioDataset(dir=dir, seq_len=15,
                           path_to_info_file=info_file_path)
    print(dataset.total_segments)
    start = timeit.default_timer()
    features = dataset.__getitem__(0)
    print(timeit.default_timer() - start)
    print(features.shape)
