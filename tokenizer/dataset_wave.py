import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Any
import os

class AudioDataset(Dataset):
    def __init__(self, dir, seq_len, sample_rate=16000, overlap_percentage=0.5, snippet_length=1) -> None:
        super().__init__()
        self.file_paths = self.get_all_files(dir)
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.overlap_percentage = overlap_percentage
        self.snippet_length = snippet_length
        self.segment_info = []
        self.total_segments = 0

        for file in self.file_paths:
            waveform = self.load_audio(file)
            snippets = self.split_audio_with_overlap(waveform)
            for i in range(len(snippets)):
                self.segment_info.append((file, i))
                self.total_segments += 1

    def __len__(self):
        return self.total_segments

    def __getitem__(self, index: Any) -> Any:
        file, snippet_idx = self.segment_info[index]
        waveform = self.load_audio(file)
        waveform_mono = self.stereo_to_mono_convertor(waveform)
        snippets = self.split_audio_with_overlap(waveform_mono)
        snippet = snippets[snippet_idx]

        # Ensure the snippet is the correct length
        if snippet.size(1) < self.seq_len:
            pad_length = self.seq_len - snippet.size(1)
            padding = torch.zeros(snippet.size(0), pad_length)
            snippet = torch.cat((snippet, padding), dim=1)
        elif snippet.size(1) > self.seq_len:
            snippet = snippet[:, :self.seq_len]

        return snippet

    def get_all_files(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform

    def stereo_to_mono_convertor(self, signal):
        # If there is more than 1 channel in your audio
        if signal.shape[0] > 1:
            # Do a mean of all channels and keep it in one channel
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def split_audio_with_overlap(self, waveform):
        snippet_length = int(self.sample_rate * self.snippet_length)
        overlap_length = int(snippet_length * self.overlap_percentage)
        step = snippet_length - overlap_length
        snippets = [waveform[:, i:i + snippet_length] for i in range(0, waveform.size(1) - snippet_length + 1, step)]
        return snippets
