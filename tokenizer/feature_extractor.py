import librosa
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    def __init__(self, sample_rate, n_mfcc):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.scaler = StandardScaler()

    def extract_features(self, waveform):
        # Ensure that waveform is 2D
        if waveform.ndim == 1:
            waveform = np.expand_dims(waveform, axis=0)

        # Extract features for each channel
        features_list = []
        for channel in waveform:
            mfccs = librosa.feature.mfcc(y=channel, sr=self.sample_rate,
                                         n_mfcc=self.n_mfcc)
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            """
            chroma = librosa.feature.chroma_stft(y=channel, sr=self.sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(y=channel,
                                                                  sr=self.sample_rate)
            tonnetz = librosa.feature.tonnetz(y=channel, sr=self.sample_rate)
            rms = librosa.feature.rms(y=channel)
            zcr = librosa.feature.zero_crossing_rate(y=channel)
            """

            features = np.concatenate((mfccs, delta_mfccs, delta2_mfccs), axis=0)
            standardized_features = self.scaler.fit_transform(features.T).T
            features_list.append(standardized_features)
        combined_features = np.stack(features_list, axis=0)
        return combined_features


if __name__ == "__main__":
    audio_file = "debussy.wav"
    waveform, sr = librosa.load(audio_file)
    extractor = FeatureExtractor(sr, 13)
    features = extractor.extract_features(waveform)
    plt.figure(figsize=(20, 8))
    librosa.display.specshow(features, x_axis="time", sr=sr)
    plt.colorbar(format="%+2f")
    plt.show()

