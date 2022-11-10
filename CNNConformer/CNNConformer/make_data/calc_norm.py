from pathlib import Path
import json
from tqdm import tqdm
import math as m
import numpy as np
from librosa.util import normalize
import torch

from CausalHiFiGAN.tools.file_io import load_list, load_json
from CausalHiFiGAN.tools.wav_io import load_wav
from CausalHiFiGAN.tools.melspectrogram import MelSpectrogramExtractor


def calc_norm(path_dir_list=Path("./data/list"),
              path_dir_param=Path("./data/param"),
              path_config_vocoder="../CausalHiFiGAN/config_v1.json"):
    print("--- calculate norm ---")

    # prepare directory

    path_dir_param.mkdir(exist_ok=1, parents=1)

    # load wav path

    list_path_wav = load_list(path_dir_list / "wav_train.txt")

    # load config

    hv = load_json(path_config_vocoder)

    # prepare mel spectrogram extractor

    f_melspe = MelSpectrogramExtractor(n_fft=hv.n_fft,
                                       win_size=hv.win_size,
                                       hop_size=hv.hop_size,
                                       sampling_rate=hv.sampling_rate,
                                       num_mels=hv.num_mels,
                                       fmin=hv.fmin,
                                       fmax=hv.fmax)

    # calculate norm

    with torch.no_grad():
        mean = 0.0
        std = 0.0
        n_frame = 0

        print(" -- calculate mean --")
        for path_wav in tqdm(list_path_wav):
            wav = load_wav(path_wav)
            wav = normalize(wav) * 0.95
            wav = torch.from_numpy(wav.astype(np.float32))
            melspe = f_melspe(wav)
            mean += torch.sum(melspe)
            n_frame += melspe.size(-1)
        mean = mean / (n_frame * hv.num_mels)

        print(f"mean = {mean}")
        print(" -- calculate standard deviation --")
        for path_wav in tqdm(list_path_wav):
            wav = load_wav(path_wav)
            wav = normalize(wav) * 0.95
            wav = torch.from_numpy(wav.astype(np.float32))
            melspe = f_melspe(wav)
            std += torch.sum((melspe - mean)**2)
        std = m.sqrt(std / (n_frame * hv.num_mels))

        print(f"std = {std}")

        dict_stats = {"mean": mean.item(),
                      "std": std}

        with open(path_dir_param / "stats.json", "w") as js:
            json.dump(dict_stats,
                      js, indent=4)
