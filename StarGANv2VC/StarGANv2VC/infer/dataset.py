import random
import numpy as np
from librosa.util import normalize
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..tools.speaker_dict import inverse_dict_speaker
from CausalHiFiGAN.tools.file_io import load_json, load_list
from CausalHiFiGAN.tools.wav_io import load_wav
from CausalHiFiGAN.tools.melspectrogram import MelSpectrogramExtractor


class SpeValidDataset(Dataset):
    def __init__(self,
                 path_dir_list_input,
                 path_dir_list_target,
                 path_stats,
                 dict_speaker,
                 hv):

        # load

        list_path_list_input = sorted((path_dir_list_input).glob("*.txt"))
        list_path_list_target = sorted((path_dir_list_target).glob("*.txt"))
        dict_num_speaker = inverse_dict_speaker(dict_speaker)

        self.list_path_wav_input = []  # for each utterance
        self.list_num_input = []  # for each utterance
        self.list_path_wav_target = []  # for each utterance
        self.list_num_target = []  # for each utterance

        for path_list in list_path_list_input:
            num_speaker = dict_num_speaker[path_list.stem]
            list_path_wav_sp = load_list(path_list)
            self.list_path_wav_input.extend(list_path_wav_sp)
            self.list_num_input.extend([num_speaker] * len(list_path_wav_sp))

        for path_list in list_path_list_target:
            num_speaker = dict_num_speaker[path_list.stem]
            list_path_wav_sp = load_list(path_list)
            self.list_path_wav_target.extend(list_path_wav_sp)
            self.list_num_target.extend([num_speaker] * len(list_path_wav_sp))

        self.len_combination = len(self.list_path_wav_input) * len(self.list_path_wav_target)
        self.len_list_input = len(self.list_path_wav_input)

        stats = load_json(path_stats)
        self.mean = stats.mean
        self.std = stats.std

        # initialize

        self.f_melspe = MelSpectrogramExtractor(n_fft=hv.n_fft,
                                                win_size=hv.win_size,
                                                hop_size=hv.hop_size,
                                                sampling_rate=hv.sampling_rate,
                                                num_mels=hv.num_mels,
                                                fmin=hv.fmin,
                                                fmax=hv.fmax)
        self.buffer_dict_target = {"spe": None, "num": -1}

    def __getitem__(self, index):
        index_input = index % self.len_list_input
        index_target = index // self.len_list_input
        num_target = self.list_num_target[index_target]

        # load wav

        path_wav_input = self.list_path_wav_input[index_input]
        wav_input = load_wav(path_wav_input)
        wav_input = normalize(wav_input) * 0.95
        wav_input = torch.from_numpy(wav_input.astype(np.float32))

        # get mel spectrogram

        spe_input = self.f_melspe(wav_input)
        spe_input = (spe_input - self.mean) / self.std

        if self.buffer_dict_target["num"] == num_target:
            spe_target = self.buffer_dict_target["spe"]
        else:
            # load wav

            path_wav_target = self.list_path_wav_target[index_target]
            wav_target = load_wav(path_wav_target)
            wav_target = normalize(wav_target) * 0.95
            wav_target = torch.from_numpy(wav_target.astype(np.float32))

            # get mel spectrogram

            spe_target = self.f_melspe(wav_target)
            spe_target = (spe_target - self.mean) / self.std

            self.buffer_dict_target = {"spe": spe_target, "num": num_target}

        return spe_input.unsqueeze(0).unsqueeze(0), spe_target.unsqueeze(0).unsqueeze(0), torch.LongTensor([num_target])

    def __len__(self):
        return self.len_combination
