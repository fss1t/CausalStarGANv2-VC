import random
import numpy as np
from librosa.util import normalize
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..tools.f0 import F0Extractor
from CausalHiFiGAN.tools.file_io import load_json, load_list
from CausalHiFiGAN.tools.wav_io import load_wav
from CausalHiFiGAN.tools.melspectrogram import MelSpectrogramExtractor


class SpeF0Dataset(Dataset):
    def __init__(self,
                 path_list_wav,
                 path_stats,
                 h,
                 hv,
                 segment_size,
                 randomize):
        if randomize:
            random.seed(h.seed)
            self.randomize = 1
        else:
            self.randomize = 0

        # load

        self.list_path_wav = load_list(path_list_wav)

        stats = load_json(path_stats)
        self.mean = stats.mean
        self.std = stats.std

        # initialize

        self.segment_size = segment_size

        self.f_melspe = MelSpectrogramExtractor(n_fft=hv.n_fft,
                                                win_size=hv.win_size,
                                                hop_size=hv.hop_size,
                                                sampling_rate=hv.sampling_rate,
                                                num_mels=hv.num_mels,
                                                fmin=hv.fmin,
                                                fmax=hv.fmax)
        self.f_f0 = F0Extractor(hv.hop_size,
                                hv.sampling_rate,
                                h.f0_floor,
                                h.f0_ceil,
                                h.f0_base)

    def __getitem__(self, index):
        # load wav

        path_wav = self.list_path_wav[index]
        wav = load_wav(path_wav)
        wav = normalize(wav) * 0.95
        wav = torch.from_numpy(wav.astype(np.float32))

        if self.randomize:
            # randomize gain

            gain = (1.0 + random.random()) / 2.0
            wav = gain * wav

        # cut segment

        if wav.size(-1) >= self.segment_size:
            if self.randomize:
                # randomize segment

                max_start_wav = wav.size(-1) - self.segment_size
                i_start_wav = random.randint(0, max_start_wav)
            else:
                i_start_wav = 0
            wav = wav[i_start_wav:i_start_wav + self.segment_size]
        else:
            wav = F.pad(wav, (0, self.segment_size - wav.size(-1)))

        # get mel spectrogram

        spe = self.f_melspe(wav)
        spe = (spe - self.mean) / self.std

        # get f0, vuv

        f0, vuv = self.f_f0(wav.numpy())
        f0 = torch.from_numpy(f0.astype(np.float32))
        vuv = torch.from_numpy(vuv.astype(np.float32))

        return spe.unsqueeze(0), f0, vuv

    def __len__(self):
        return len(self.list_path_wav)
