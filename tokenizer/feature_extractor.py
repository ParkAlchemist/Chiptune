import librosa
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import functools

from memory_profiler import profile

from utils import Feature


class FeatureExtractor:
    def __init__(self, sample_rate, features_to_extract, target_shape, n_mfcc=13):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

        self.n_fft = 1024
        self.fmin = 125
        self.fmax = 7600
        self.nmels = 80
        self.hop_len = 1024//4
        self.spec_power = 1

        self.features_to_extract = features_to_extract
        self.target_shape = target_shape
        self.scalers = {feature: StandardScaler() for feature in features_to_extract}
        self.pca = PCA(n_components=0.99)
        self.n_fft = 1024
        self.func_list = []
        self.mel_basis = librosa.filters.mel(sr=self.sample_rate,
                                             n_fft=self.n_fft, fmin=self.fmin,
                                             fmax=self.fmax, n_mels=self.nmels)

        self.feature_methods = {
            Feature.MEL_SPECTROGRAM: self.get_mel_spec,
            Feature.MFCC: self.get_mfcc,
            Feature.CHROMA: self.get_chroma,
            Feature.CHROMA_CQT: self.get_chroma_cqt,
            Feature.SPECTRAL_CENTROID: self.get_spectral_centroid,
            Feature.SPECTRAL_CONTRAST: self.get_spectral_contrast,
            Feature.SPECTRAL_FLATNESS: self.get_spectral_flatness,
            Feature.SPECTRAL_ROLLOFF: self.get_spectral_rolloff,
            Feature.ONSET: functools.partial(self.get_onset, tempo=False),
            Feature.RMS: self.get_rms,
            Feature.ZCR: self.get_zcr
        }

        if Feature.ONSET in features_to_extract and Feature.TEMPO in features_to_extract:
            self.feature_methods[Feature.ONSET] = functools.partial(self.get_onset, tempo=True)

    @staticmethod
    def pad_or_truncate(feature, target_shape):
        if feature.shape[1] < target_shape[1]:
            # Pad with zeros
            padding = np.zeros(
                (feature.shape[0], target_shape[1] - feature.shape[1]))
            feature = np.concatenate((feature, padding), axis=1)
        elif feature.shape[1] > target_shape[1]:
            # Truncate
            feature = feature[:, :target_shape[1]]
        return feature

    @staticmethod
    def aggregate_features(features, window_length, hop_length):
        aggregated_features = []
        for i in range(0, features.shape[1], hop_length):
            window_features = features[:, i:i + window_length]
            mean_features = np.mean(window_features, axis=1)
            var_features = np.var(window_features, axis=1)
            aggregated_features.append(np.concatenate((mean_features, var_features)))
        return np.array(aggregated_features).T

    def get_mel_spec(self, waveform):
        spec = np.abs(librosa.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_len)) ** self.spec_power
        mel_spec = np.dot(self.mel_basis, spec)
        return mel_spec

    def get_mfcc(self, waveform):
        mfccs = librosa.feature.mfcc(y=waveform, sr=self.sample_rate,
                                     n_mfcc=self.n_mfcc)
        mfccs = self.scalers[Feature.MFCC].fit_transform(mfccs.T).T

        return mfccs

    def get_chroma(self, waveform):
        chroma = librosa.feature.chroma_stft(y=waveform,
                                             sr=self.sample_rate,
                                             n_fft=self.n_fft)
        chroma = self.scalers[Feature.CHROMA].fit_transform(chroma.T).T
        return chroma

    def get_chroma_cqt(self, waveform):
        chroma_cqt = librosa.feature.chroma_cqt(y=waveform,
                                                sr=self.sample_rate)
        chroma_cqt = self.scalers[Feature.CHROMA_CQT].fit_transform(
            chroma_cqt.T).T
        return chroma_cqt

    def get_spectral_centroid(self, waveform):
        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform, sr=self.sample_rate)
        spectral_centroid = self.scalers[
            Feature.SPECTRAL_CENTROID].fit_transform(spectral_centroid.T).T
        return spectral_centroid

    def get_spectral_contrast(self, waveform):
        spectral_contrast = librosa.feature.spectral_contrast(y=waveform,
                                                              sr=self.sample_rate)
        spectral_contrast = self.scalers[
            Feature.SPECTRAL_CONTRAST].fit_transform(spectral_contrast.T).T
        return spectral_contrast

    def get_spectral_flatness(self, waveform):
        spectral_flatness = librosa.feature.spectral_flatness(y=waveform)
        spectral_flatness = self.scalers[
            Feature.SPECTRAL_FLATNESS].fit_transform(spectral_flatness.T).T
        return spectral_flatness

    def get_spectral_rolloff(self, waveform):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform,
                                                            sr=self.sample_rate)
        spectral_rolloff = self.scalers[
            Feature.SPECTRAL_ROLLOFF].fit_transform(spectral_rolloff.T).T
        return spectral_rolloff

    def get_onset(self, waveform, tempo = False):
        onset_env = librosa.onset.onset_strength(y=waveform,
                                                 sr=self.sample_rate)
        if tempo:
            tempo, beat_frames = librosa.beat.beat_track(y=waveform,
                                                         sr=self.sample_rate,
                                                         onset_envelope=onset_env,
                                                         hop_length=512)
            one_hot = np.array([1 if i in beat_frames else 0 for i in range(onset_env.shape[0])]).reshape(1,onset_env.shape[0])
        onset_env = self.scalers[Feature.ONSET].fit_transform(onset_env.reshape(-1, 1)).T
        return np.vstack((onset_env, one_hot)) if tempo else onset_env

    def get_rms(self, waveform):
        rms = librosa.feature.rms(y=waveform)
        rms = self.scalers[Feature.RMS].fit_transform(rms.T).T
        return rms

    def get_zcr(self, waveform):
        zcr = librosa.feature.zero_crossing_rate(y=waveform)
        zcr = self.scalers[Feature.ZCR].fit_transform(zcr.T).T
        return zcr


    def extract_features(self, waveform):

        # Extract features for each channel
        features_list = []
        for channel in waveform:
            feature_set = [self.feature_methods[feature](channel) for feature in self.features_to_extract]

            # Concatenate all selected features
            features = np.concatenate(feature_set, axis=0)
            #padded_features = self.pad_or_truncate(standardized_features,
            #                                       self.target_shape)

            # Apply temporal aggregation
            """
            window_length = 10
            hop_length = window_length // 2
            aggregated_features = self.aggregate_features(features, window_length, hop_length)
            """

            features_list.append(features)

        combined_features = np.stack(features_list, axis=0)
        """
        combined_features = (self.pca.fit_transform(combined_features
                                                   .reshape(-1,combined_features.shape[-1]))
                             .reshape(combined_features.shape[0], -1, self.pca.n_components_))
        """
        return combined_features.astype(np.float32)


def plot_combined_features(waveform, sr, features_to_extract):
    plt.figure(figsize=(20, 16))

    if 'mel_spectrogram' in features_to_extract:
        plt.subplot(3, 2, 1)
        S = librosa.feature.melspectrogram(y=waveform, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-Spectrogram')

    if 'chroma' in features_to_extract:
        plt.subplot(3, 2, 2)
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
        plt.colorbar()
        plt.title('Chroma Features')

    if 'cqt' in features_to_extract:
        plt.subplot(3, 2, 3)
        C = librosa.cqt(y=waveform, sr=sr)
        C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        librosa.display.specshow(C_dB, sr=sr, x_axis='time', y_axis='cqt_note')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q Transform')

    if 'mfcc' in features_to_extract:
        plt.subplot(3, 2, 4)
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title('MFCCs')

    if 'spectral_centroid' in features_to_extract:
        plt.subplot(3, 2, 5)
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        frames = range(len(spectral_centroid))
        t = librosa.frames_to_time(frames, sr=sr)
        plt.plot(t, spectral_centroid, color='r')
        plt.title('Spectral Centroid')

    if 'spectral_rolloff' in features_to_extract:
        plt.subplot(3, 2, 6)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        frames = range(len(spectral_rolloff))
        t = librosa.frames_to_time(frames, sr=sr)
        plt.plot(t, spectral_rolloff, color='b')
        plt.title('Spectral Rolloff')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    audio_file = "debussy.wav"
    waveform, sr = librosa.load(audio_file)
    features_to_extract = [Feature.MEL_SPECTROGRAM]  # Specify the features you want to extract
    extractor = FeatureExtractor(sr, features_to_extract, target_shape=(256, 200))
    features = extractor.extract_features(np.stack((waveform, waveform), axis=0))
    print(features.shape)
    plot_combined_features(waveform, sr, ['mel_spectrogram', 'chroma', 'mfcc'])
