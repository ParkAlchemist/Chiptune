import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Any
import os


class AudioDataset(Dataset):
    def __init__(self, dir, seq_len, n_mels=128, sample_rate = 16000, overlap_percentage = 0.5, snippet_length = 15) -> None:
        super().__init__()
        self.file_paths = self.get_all_files(dir)
        self.n_mels = n_mels
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.overlap_percentage = overlap_percentage
        self.snippet_length = snippet_length

        self.segment_info = []
        self.total_segments = 0

        for file in self.file_paths:
            waveform = self.load_audio(file)
            snippets = self.split_audio_with_overlap(waveform)
            for snippet in snippets:
                self.segment_info.append((file, snippet))
                self.total_segments += 1

    def __len__(self):
        return self.total_segments

    def __getitem__(self, index: Any) -> Any:

        file, snippet = self.segment_info[index]
        spectrogram = self.audio_to_spectrogram(snippet)
        return spectrogram

    def get_all_files(self, directory):
        return [os.path.join(directory, f) for f in os.listdir(directory) if
                os.path.isfile(os.path.join(directory, f))]

    def load_audio(self, file_path):
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(
                waveform)
        return waveform

    def split_audio_with_overlap(self, waveform):
        snippet_length = int(self.sample_rate * self.snippet_length)
        overlap_length = int(snippet_length * self.overlap_percentage)
        step = snippet_length - overlap_length
        snippets = [waveform[:, i:i + snippet_length] for i in
                    range(0, waveform.size(1) - snippet_length + 1, step)]
        return snippets

    def audio_to_spectrogram(self, waveform):
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=4096,
            hop_length=512,
            n_mels=self.n_mels
        )(waveform)
        spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        normalized_spectrogram = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())

        # Padding
        if normalized_spectrogram.size(2) < self.seq_len:
            pad_length = self.seq_len - normalized_spectrogram.size(2)
            padding = torch.zeros(normalized_spectrogram.size(0),
                                  normalized_spectrogram.size(1), pad_length)
            normalized_spectrogram = torch.cat(
                (normalized_spectrogram, padding), dim=2)

        # Truncating
        if normalized_spectrogram.size(2) > self.seq_len:
            normalized_spectrogram = normalized_spectrogram[:, :,
                                     :self.seq_len]

        return normalized_spectrogram
