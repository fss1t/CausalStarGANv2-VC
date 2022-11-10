import random
import numpy as np
from librosa.util import normalize
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from CausalHiFiGAN.tools.file_io import load_json, load_list
from CausalHiFiGAN.tools.wav_io import load_wav
from CausalHiFiGAN.tools.melspectrogram import MelSpectrogramExtractor


class SpeLabDataset(Dataset):
    def __init__(self,
                 path_list_wav,
                 path_list_lab,
                 path_stats,
                 path_phoneme,
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
        self.list_path_lab = load_list(path_list_lab)

        stats = load_json(path_stats)
        self.mean = stats.mean
        self.std = stats.std

        self.dict_phoneme = load_json(path_phoneme)

        # initialize

        self.sampling_rate = hv.sampling_rate
        self.hop_size_lab = hv.hop_size * 4
        self.segment_size = segment_size
        self.len_lab = segment_size // self.hop_size_lab

        self.f_melspe = MelSpectrogramExtractor(n_fft=hv.n_fft,
                                                win_size=hv.win_size,
                                                hop_size=hv.hop_size,
                                                sampling_rate=hv.sampling_rate,
                                                num_mels=hv.num_mels,
                                                fmin=hv.fmin,
                                                fmax=hv.fmax)

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

        i_start_wav = 0
        if wav.size(-1) >= self.segment_size:
            if self.randomize:
                # randomize segment

                max_start_wav = wav.size(-1) - self.segment_size
                i_start_wav = random.randint(0, max_start_wav)
            wav = wav[i_start_wav:i_start_wav + self.segment_size]
        else:
            wav = F.pad(wav, (0, self.segment_size - wav.size(-1)))

        # get mel spectrogram

        spe = self.f_melspe(wav)
        spe = (spe - self.mean) / self.std

        # load lab

        path_lab = self.list_path_lab[index]
        with open(path_lab, "r") as labf:
            lines_lab = labf.read().splitlines()

        # encode lab

        lab = torch.zeros(self.len_lab, dtype=torch.long)
        i = 0
        for line in lines_lab:
            items = line.split()
            time_start_phoneme, time_end_phoneme, phoneme = float(items[0]), float(items[1]), items[2]
            i_start_phoneme = (self.sampling_rate * time_start_phoneme - i_start_wav) / self.hop_size_lab
            i_end_phoneme = (self.sampling_rate * time_end_phoneme - i_start_wav) / self.hop_size_lab
            code = self.dict_phoneme[phoneme]
            while(i_start_phoneme <= i + 0.5 and i + 0.5 < i_end_phoneme and i < self.len_lab):
                lab[i] = code
                i += 1

        return spe.unsqueeze(0), lab

    def __len__(self):
        return len(self.list_path_wav)
