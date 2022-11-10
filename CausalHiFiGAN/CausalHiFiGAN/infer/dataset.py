import random
import numpy as np
from librosa.util import normalize
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from CausalHiFiGAN.tools.file_io import load_json, load_list
from CausalHiFiGAN.tools.wav_io import load_wav
from CausalHiFiGAN.tools.melspectrogram import MelSpectrogramExtractor


class MelValidDataset(Dataset):
    def __init__(self,
                 path_list,
                 h,
                 tail_size):
        # load wav path list

        self.list_path_wav = load_list(path_list)

        # initialize

        self.pad = (h.n_fft - h.hop_size) // 2
        self.tail_size = tail_size

        self.f_melspe = MelSpectrogramExtractor(n_fft=h.n_fft,
                                                win_size=h.win_size,
                                                hop_size=h.hop_size,
                                                sampling_rate=h.sampling_rate,
                                                num_mels=h.num_mels,
                                                fmin=h.fmin,
                                                fmax=h.fmax,
                                                padding=False)

    def __getitem__(self, index):
        path_wav = self.list_path_wav[index]

        # load

        wav = load_wav(path_wav)
        wav = normalize(wav) * 0.95
        wav = torch.from_numpy(wav.astype(np.float32))

        wav_pad = F.pad(wav, (self.tail_size + self.pad, self.pad))

        # get mel spectrogram

        spe = self.f_melspe(wav_pad)

        return spe.unsqueeze(0)

    def __len__(self):
        return len(self.list_path_wav)
