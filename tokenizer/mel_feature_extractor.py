import librosa
import librosa.feature
import librosa.display
import numpy as np
import torch


class FeatureExtractor:
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

        self.n_fft = 1024
        self.nmels = 128
        self.hoplen = 256
        self.tgt_len = 1288

    def get_mel_spec(self, waveform):
        mel_spectrogram = librosa.feature.melspectrogram(y=waveform,
                                                         sr=self.sample_rate,
                                                         n_mels=self.nmels,
                                                         hop_length=self.hoplen)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        if log_mel_spectrogram.shape[2] < self.tgt_len:
            pad_length = self.tgt_len - log_mel_spectrogram.shape[2]
            padding = np.zeros(log_mel_spectrogram.shape[0],
                                  log_mel_spectrogram.shape[1], pad_length)
            log_mel_spectrogram = np.concatenate((log_mel_spectrogram, padding), axis=2)
        elif log_mel_spectrogram.shape[2] > self.tgt_len:
            log_mel_spectrogram = log_mel_spectrogram[:, :, :self.tgt_len]

        mean = np.mean(log_mel_spectrogram)
        std = np.std(log_mel_spectrogram)
        standardized_mel_spec = (log_mel_spectrogram - mean) / std

        return standardized_mel_spec


if __name__ == "__main__":
    audio_file = "../../debussy.wav"
    waveform, sr = librosa.load(audio_file)
    extractor = FeatureExtractor(sample_rate=sr)
    features = extractor.get_mel_spec(waveform)
    print(features.shape)
