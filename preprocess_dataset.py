import os
import torchaudio
import torch
from config_audio import get_config


def trim_silence(waveform, sample_rate, threshold=0.001, padding=0.1):
    energy = waveform.pow(2).mean(dim=0)
    non_silent_indices = (energy > threshold).nonzero(as_tuple=True)[0]

    if len(non_silent_indices) == 0:
        return waveform  # Return the original waveform if no non-silent parts are found

    start_index = max(0, non_silent_indices[0] - int(padding * sample_rate))
    end_index = min(waveform.size(1),
                    non_silent_indices[-1] + int(padding * sample_rate))

    return waveform[:, start_index:end_index]


def pad_waveform(waveform, target_length):
    padding_length = target_length - waveform.size(1)
    if padding_length > 0:
        padding = torch.zeros((waveform.size(0), padding_length))
        waveform = torch.cat((waveform, padding), dim=1)
    return waveform


def preprocess_audio_files(orig_directory, chip_directory, segment_length, sample_rate):
    for orig_file, chip_file in zip(sorted(os.listdir(orig_directory)), sorted(os.listdir(chip_directory))):
        if orig_file.endswith('.mp3') and chip_file.endswith('.mp3'):
            orig_path = os.path.join(orig_directory, orig_file)
            chip_path = os.path.join(chip_directory, chip_file)

            orig_waveform, sr = torchaudio.load(orig_path)
            chip_waveform, sr_chip = torchaudio.load(chip_path)

            if sr != sample_rate:
                orig_waveform = torchaudio.transforms.Resample(sr, sample_rate)(orig_waveform)
            if sr_chip != sample_rate:
                chip_waveform = torchaudio.transforms.Resample(sr_chip, sample_rate)(chip_waveform)

            orig_waveform = trim_silence(orig_waveform, sample_rate)
            chip_waveform = trim_silence(chip_waveform, sample_rate)

            max_length = max(orig_waveform.size(1), chip_waveform.size(1))
            target_length = (max_length + segment_length - 1) // segment_length * segment_length

            orig_waveform = pad_waveform(orig_waveform, target_length)
            chip_waveform = pad_waveform(chip_waveform, target_length)

            torchaudio.save(orig_path, orig_waveform, sample_rate)
            torchaudio.save(chip_path, chip_waveform, sample_rate)
            print(f'Processed {orig_file} and {chip_file}')


if __name__ == "__main__":
    orig_path = "dataset/orig"
    chip_path = "dataset/chip"

    config_audio = get_config()
    segment_length = config_audio["seg_len"]
    sample_rate = config_audio["sr"]

    preprocess_audio_files(orig_path, chip_path, segment_length, sample_rate)
