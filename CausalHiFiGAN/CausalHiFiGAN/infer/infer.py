import sys
from importlib import import_module
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pathlib import Path
from tqdm import tqdm
import torch

from .dataset import MelValidDataset
from ..models.generator import Generator

from CausalHiFiGAN.tools.file_io import load_json, load_list
from CausalHiFiGAN.tools.wav_io import write_wav

torch.backends.cudnn.benchmark = True


def infer(path_dir_list=Path("./data/list"),
          path_checkpoint=Path("./checkpoint/g_01000000"),
          path_result=Path("./result"),
          path_config="./config_v1.json"):
    print("--- infer ---")

   # prepare directory

    path_result.mkdir(exist_ok=1)

    # load config

    h = load_json(path_config)

    device = torch.device(h.device)

    # prepare model

    vocoder = Generator(h).to(device)

    assert path_checkpoint.exists()
    cp = torch.load(path_checkpoint, map_location=lambda storage, loc: storage)
    vocoder.load_state_dict(cp["generator"])
    print(f"loaded {path_checkpoint}")

    vocoder.eval().remove_weight_norm()

    # prepare dataset

    dataset = MelValidDataset(path_dir_list / "valid.txt",
                              h,
                              vocoder.tail)

    # infer
    print(f" -- inference --")

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataset)):
            spe = batch.to(device)

            wav_output = vocoder(spe)

            path_wav_input = dataset.list_path_wav[i]
            name_output = f"{path_wav_input.parent.stem}_{path_wav_input.stem}"
            write_wav(path_result / f"{name_output}.wav", wav_output, h.sampling_rate)
