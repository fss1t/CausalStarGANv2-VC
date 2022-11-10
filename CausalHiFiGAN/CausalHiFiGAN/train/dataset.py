from pathlib import Path
import random
import numpy as np
from librosa.util import normalize
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..tools.file_io import load_list
from ..tools.wav_io import load_wav
from ..tools.melspectrogram import MelSpectrogramExtractor


class MelDataset(Dataset):
    def __init__(self,
                 path_list,
                 h,
                 segment_size,
                 tail_size,
                 randomize):
        if randomize:
            random.seed(h.seed)
            self.randomize = 1
        else:
            self.randomize = 0

        # load wav path list

        self.list_path_wav = load_list(path_list)

        # initialize

        self.segment_size = segment_size
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

        if self.randomize:
            # randomize gain

            gain = (1.0 + random.random()) / 2.0
            wav = gain * wav

        # cut segment

        wav_pad = F.pad(wav, (self.tail_size + self.pad, self.pad))
        if wav.size(-1) >= self.segment_size:
            if self.randomize:
                # randomize segment

                max_start_wav = wav.size(-1) - self.segment_size
                i_start_wav = random.randint(0, max_start_wav)
            else:
                i_start_wav = 0
            wav = wav[i_start_wav:i_start_wav + self.segment_size]
            wav_pad = wav_pad[i_start_wav:i_start_wav + self.segment_size + self.tail_size + self.pad * 2]
        else:
            wav = F.pad(wav, (0, self.segment_size - wav.size(-1)))
            wav_pad = F.pad(wav_pad, (0, self.segment_size + self.tail_size + self.pad * 2 - wav_pad.size(-1)))

        # get mel spectrogram

        spe = self.f_melspe(wav_pad)

        return spe, wav

    def __len__(self):
        return len(self.list_path_wav)
