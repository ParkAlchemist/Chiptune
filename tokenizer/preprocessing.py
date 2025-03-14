import os
import torch

from model.encoder import get_encoder
from model.quantizer import get_quantizer
from config import get_enc_weights_file_path, get_quant_weights_file_path, get_config
from utils import get_loaders

from dataset_wave import AudioDataset


def write_dataset_info(appendix: str):
    """
    Writes dataset info to .txt file.
    For each segment writes filepath, snippet index and original sample_rate
    :param appendix: orig or chip
    :return: None
    """
    assert appendix == 'orig' or appendix == 'chip', "Invalid appendix, use orig or chip"
    os.chdir("../dataset")
    config = get_config()
    dataset = AudioDataset(dir=f"../dataset/{appendix}", seq_len=config["model_params"]["seq_len"])
    with open(f"dataset_info_{appendix}.txt", "w") as f:
        segment_data = [f"{file_name} | {idx} | {sr}\n" for (file_name, idx, sr) in dataset.segment_info]
        f.writelines(segment_data)


def init_quantizer(appendix: str, epoch_num: int):

    assert appendix == 'orig' or appendix == 'chip', "Invalid appendix, use orig or chip"
    assert epoch_num > 0, "Invalid epoch_num, value must be greater than zero"

    config = get_config()

    # Get filepaths
    enc_file_path = get_enc_weights_file_path(config, f"{epoch_num:02d}", appendix)
    quant_file_path = get_quant_weights_file_path(config, f"{epoch_num:02d}", appendix)
    data_path = "../dataset/" + appendix

    # Init classes
    enc = get_encoder(config)
    quant = get_quantizer(config)

    # Load pretrained encoder
    enc_state = torch.load(enc_file_path)
    enc.load_state_dict(enc_state["encoder_state_dict"])

    # Init dataset and data loaders
    dataset = AudioDataset(data_path, config["model_params"]["seq_len"], path_to_info_file=f"../dataset/dataset_info_{appendix}.txt")
    train_loader, _ = get_loaders(dataset,
                                  config["training_params"]["shuffle_dataset"],
                                  config["training_params"]["validation_split"],
                                  config["training_params"]["batch_size"])

    print("Initializing codebook...")
    quant.initialize_codebook(enc, train_loader)
    print("Saving quantizer...")
    torch.save({
        "quantizer_state_dict": quant.state_dict()
    }, quant_file_path)
    print(f"Quantizer saved to {quant_file_path}")


if __name__ == "__main__":
    appendix = "chip"
    #write_dataset_info(appendix)
    init_quantizer(appendix=appendix, epoch_num=10)
