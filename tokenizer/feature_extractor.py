import librosa
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from memory_profiler import profile

from utils import Feature


class FeatureExtractor:
    def __init__(self, sample_rate, features_to_extract, target_shape, n_mfcc=13):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.features_to_extract = features_to_extract
        self.target_shape = target_shape
        self.scalers = {feature: StandardScaler() for feature in features_to_extract}
        self.pca = PCA(n_components=0.99)
        self.n_fft = 1024

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

    def extract_features(self, waveform):

        # Extract features for each channel
        features_list = []
        for channel in waveform:
            feature_set = []

            if Feature.MFCC in self.features_to_extract:
                mfccs = librosa.feature.mfcc(y=channel, sr=self.sample_rate,
                                             n_mfcc=self.n_mfcc)
                mfccs = self.scalers[Feature.MFCC].fit_transform(mfccs.T).T
                feature_set.append(mfccs)

                if Feature.MFCC_DELTA in self.features_to_extract:
                    delta_mfccs = librosa.feature.delta(mfccs)
                    delta_mfccs = self.scalers[Feature.MFCC_DELTA].fit_transform(delta_mfccs.T).T
                    feature_set.append(delta_mfccs)

                if Feature.MFCC_DELTA_DELTA in self.features_to_extract:
                    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                    delta2_mfccs = self.scalers[Feature.MFCC_DELTA_DELTA].fit_transform(delta2_mfccs.T).T
                    feature_set.append(delta2_mfccs)


            if Feature.MEL_SPECTROGRAM in self.features_to_extract:
                mel_spectrogram = librosa.feature.melspectrogram(y=channel,
                                                                 sr=self.sample_rate)
                log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                log_mel_spectrogram = self.scalers[Feature.MEL_SPECTROGRAM].fit_transform(log_mel_spectrogram.T).T
                feature_set.append(log_mel_spectrogram)

            if Feature.CQT in self.features_to_extract:
                cqt = librosa.cqt(y=channel, sr=self.sample_rate)
                cqt = self.scalers[Feature.CQT].fit_transform(cqt.T).T
                feature_set.append(cqt)

            if Feature.CHROMA in self.features_to_extract:
                chroma = librosa.feature.chroma_stft(y=channel,
                                                     sr=self.sample_rate, n_fft=self.n_fft)
                chroma = self.scalers[Feature.CHROMA].fit_transform(chroma.T).T
                feature_set.append(chroma)

            if Feature.CHROMA_CQT in self.features_to_extract:
                chroma_cqt = librosa.feature.chroma_cqt(y=channel,
                                                 sr=self.sample_rate)
                chroma_cqt = self.scalers[Feature.CHROMA_CQT].fit_transform(chroma_cqt.T).T
                feature_set.append(chroma_cqt)

            if Feature.SPECTRAL_CENTROID in self.features_to_extract:
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=channel, sr=self.sample_rate)
                spectral_centroid = self.scalers[Feature.SPECTRAL_CENTROID].fit_transform(spectral_centroid.T).T
                feature_set.append(spectral_centroid)

            if Feature.SPECTRAL_CONTRAST in self.features_to_extract:
                spectral_contrast = librosa.feature.spectral_contrast(y=channel,
                                                                    sr=self.sample_rate)
                spectral_contrast = self.scalers[Feature.SPECTRAL_CONTRAST].fit_transform(spectral_contrast.T).T
                feature_set.append(spectral_contrast)

            if Feature.SPECTRAL_FLATNESS in self.features_to_extract:
                spectral_flatness = librosa.feature.spectral_flatness(y=channel)
                spectral_flatness = self.scalers[Feature.SPECTRAL_FLATNESS].fit_transform(spectral_flatness.T).T
                feature_set.append(spectral_flatness)

            if Feature.SPECTRAL_ROLLOFF in self.features_to_extract:
                spectral_rolloff = librosa.feature.spectral_rolloff(y=channel, sr=self.sample_rate)
                spectral_rolloff = self.scalers[Feature.SPECTRAL_ROLLOFF].fit_transform(spectral_rolloff.T).T
                feature_set.append(spectral_rolloff)

            if Feature.ONSET in self.features_to_extract:
                onset_env = librosa.onset.onset_strength(y=channel,
                                                         sr=self.sample_rate)
                if Feature.TEMPO in self.features_to_extract:
                    tempo, beat_frames = librosa.beat.beat_track(y=channel,
                                                                 sr=self.sample_rate,
                                                                 onset_envelope=onset_env,
                                                                 hop_length=512)
                    #beat_times = librosa.frames_to_time(beat_frames,
                    #                                    sr=self.sample_rate,
                    #                                    hop_length=512)

                    #beat_times = self.scalers['tempo'].fit_transform(
                    #    beat_times.reshape(-1, 1)).T

                    one_hot = np.array([1 if i in beat_frames else 0 for i in range(onset_env.shape[0])]).reshape(1, onset_env.shape[0])

                    feature_set.append(one_hot)

                onset_env = self.scalers[Feature.ONSET].fit_transform(onset_env.reshape(-1, 1)).T
                feature_set.append(onset_env)

            if Feature.HPSS_HARMONIC in self.features_to_extract and Feature.HPSS_PERCUSSIVE in self.features_to_extract:
                harmonic, percussive = librosa.effects.hpss(y=channel, hop_length=512)
                harmonic = self.scalers[Feature.HPSS_HARMONIC].fit_transform(harmonic.reshape(-1, 1)).T
                percussive = self.scalers[Feature.HPSS_PERCUSSIVE].fit_transform(percussive.reshape(-1, 1)).T
                feature_set.append(harmonic)
                feature_set.append(percussive)

            if Feature.RMS in self.features_to_extract:
                rms = librosa.feature.rms(y=channel)
                rms = self.scalers[Feature.RMS].fit_transform(rms.T).T
                feature_set.append(rms)

            if Feature.ZCR in self.features_to_extract:
                zcr = librosa.feature.zero_crossing_rate(y=channel)
                zcr = self.scalers[Feature.ZCR].fit_transform(zcr.T).T
                feature_set.append(zcr)

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
    features_to_extract = ['mfcc', 'mel_spectrogram',
                           'chroma', 'chroma_cqt', 'spectral_centroid',
                           'spectral_contrast', 'spectral_flatness',
                           'spectral_rolloff', 'rms', 'zcr', 'onset', 'tempo']  # Specify the features you want to extract
    extractor = FeatureExtractor(sr, features_to_extract, target_shape=(256, 200))
    features = extractor.extract_features(np.stack((waveform, waveform), axis=0))
    print(features.shape)
    plot_combined_features(waveform, sr, ['mel_spectrogram', 'chroma', 'mfcc'])
