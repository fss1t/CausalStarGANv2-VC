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


class SpeDataset(Dataset):
    def __init__(self,
                 path_dir_list,
                 path_stats,
                 dict_speaker,
                 h,
                 hv,
                 segment_size,
                 tail_size,
                 randomize):
        if randomize:
            random.seed(h.seed)
            self.randomize = 1
        else:
            self.randomize = 0

        # load

        list_path_list_input = sorted((path_dir_list).glob("*.txt"))
        dict_num_speaker = inverse_dict_speaker(dict_speaker)

        self.list_path_wav = []  # for each utterance
        self.list_num_speaker = []  # for each utterance
        self.list_range_speaker = []  # for each speaker

        for path_list in list_path_list_input:
            num_speaker = dict_num_speaker[path_list.stem]
            list_path_wav_sp = load_list(path_list)
            self.list_path_wav.extend(list_path_wav_sp)
            self.list_num_speaker.extend([num_speaker] * len(list_path_wav_sp))
            self.list_range_speaker.append([len(self.list_path_wav) + len(list_path_wav_sp), len(self.list_path_wav)])
        self.len_list = len(self.list_path_wav)
        for range_speaker in self.list_range_speaker:
            range_speaker[1] += self.len_list

        stats = load_json(path_stats)
        self.mean = stats.mean
        self.std = stats.std

        # initialize

        self.segment_size = segment_size
        self.pad = (hv.n_fft - hv.hop_size) // 2
        self.tail_size = tail_size

        self.f_melspe_nopad = MelSpectrogramExtractor(n_fft=hv.n_fft,
                                                      win_size=hv.win_size,
                                                      hop_size=hv.hop_size,
                                                      sampling_rate=hv.sampling_rate,
                                                      num_mels=hv.num_mels,
                                                      fmin=hv.fmin,
                                                      fmax=hv.fmax,
                                                      padding=False)
        self.f_melspe = MelSpectrogramExtractor(n_fft=hv.n_fft,
                                                win_size=hv.win_size,
                                                hop_size=hv.hop_size,
                                                sampling_rate=hv.sampling_rate,
                                                num_mels=hv.num_mels,
                                                fmin=hv.fmin,
                                                fmax=hv.fmax)

    def __getitem__(self, index_input):
        num_input = self.list_num_speaker[index_input]

        # select target utterance

        index_target = random.randrange(*self.list_range_speaker[num_input]) % self.len_list
        num_target = self.list_num_speaker[index_target]

        # load wav

        path_wav_input = self.list_path_wav[index_input]
        wav_input = load_wav(path_wav_input)
        wav_input = normalize(wav_input) * 0.95
        wav_input = torch.from_numpy(wav_input.astype(np.float32))

        path_wav_target = self.list_path_wav[index_target]
        wav_target = load_wav(path_wav_target)
        wav_target = normalize(wav_target) * 0.95
        wav_target = torch.from_numpy(wav_target.astype(np.float32))

        if self.randomize:
            # randomize gain

            gain = (1.0 + random.random()) / 2.0
            wav_input = gain * wav_input
            wav_target = gain * wav_target

        # cut segment

        wav_input_pad = F.pad(wav_input, (self.tail_size + self.pad, self.pad))
        if wav_input.size(-1) >= self.segment_size:
            if self.randomize:
                # randomize segment

                max_start_wav = wav_input.size(-1) - self.segment_size
                i_start_wav = random.randint(0, max_start_wav)
            else:
                i_start_wav = 0
            wav_input_pad = wav_input_pad[i_start_wav:i_start_wav + self.segment_size + self.tail_size + self.pad * 2]
        else:
            wav_input_pad = F.pad(wav_input_pad, (0, self.segment_size + self.tail_size + self.pad * 2 - wav_input_pad.size(-1)))

        if wav_target.size(-1) >= self.segment_size:
            if self.randomize:
                # randomize segment

                max_start_wav = wav_target.size(-1) - self.segment_size
                i_start_wav = random.randint(0, max_start_wav)
            else:
                i_start_wav = 0
            wav_target = wav_target[i_start_wav:i_start_wav + self.segment_size]
        else:
            wav_target = F.pad(wav_target, (0, self.segment_size - wav_target.size(-1)))

        # get mel spectrogram

        spe_input = self.f_melspe_nopad(wav_input_pad)
        spe_input = (spe_input - self.mean) / self.std

        spe_target = self.f_melspe(wav_target)
        spe_target = (spe_target - self.mean) / self.std

        return spe_input.unsqueeze(0), num_input, spe_target.unsqueeze(0), num_target

    def __len__(self):
        return self.len_list


class SpeValidDataset(Dataset):
    def __init__(self,
                 path_dir_list_input,
                 path_dir_list_target,
                 path_stats,
                 dict_speaker,
                 h,
                 hv,
                 segment_size,
                 tail_size,
                 randomize):
        if randomize:
            random.seed(h.seed)
            self.randomize = 1
        else:
            self.randomize = 0

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

        self.segment_size = segment_size
        self.pad = (hv.n_fft - hv.hop_size) // 2
        self.tail_size = tail_size

        self.f_melspe_nopad = MelSpectrogramExtractor(n_fft=hv.n_fft,
                                                      win_size=hv.win_size,
                                                      hop_size=hv.hop_size,
                                                      sampling_rate=hv.sampling_rate,
                                                      num_mels=hv.num_mels,
                                                      fmin=hv.fmin,
                                                      fmax=hv.fmax,
                                                      padding=False)
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
        num_input = self.list_num_input[index_input]
        num_target = self.list_num_target[index_target]

        # load wav

        path_wav_input = self.list_path_wav_input[index_input]
        wav_input = load_wav(path_wav_input)
        wav_input = normalize(wav_input) * 0.95
        wav_input = torch.from_numpy(wav_input.astype(np.float32))

        if self.randomize:
            # randomize gain

            gain = (1.0 + random.random()) / 2.0
            wav_input = gain * wav_input

        # cut segment

        wav_input_pad = F.pad(wav_input, (self.tail_size + self.pad, self.pad))
        if wav_input.size(-1) >= self.segment_size:
            if self.randomize:
                # randomize segment

                max_start_wav = wav_input.size(-1) - self.segment_size
                i_start_wav = random.randint(0, max_start_wav)
            else:
                i_start_wav = 0
            wav_input_pad = wav_input_pad[i_start_wav:i_start_wav + self.segment_size + self.tail_size + self.pad * 2]
        else:
            wav_input_pad = F.pad(wav_input_pad, (0, self.segment_size + self.tail_size + self.pad * 2 - wav_input_pad.size(-1)))

        # get mel spectrogram

        spe_input = self.f_melspe_nopad(wav_input_pad)
        spe_input = (spe_input - self.mean) / self.std

        if self.buffer_dict_target["num"] == num_target:
            spe_target = self.buffer_dict_target["spe"]
        else:
            # load wav

            path_wav_target = self.list_path_wav_target[index_target]
            wav_target = load_wav(path_wav_target)
            wav_target = normalize(wav_target) * 0.95
            wav_target = torch.from_numpy(wav_target.astype(np.float32))

            # cut segment

            if wav_target.size(-1) >= self.segment_size:
                if self.randomize:
                    # randomize segment

                    max_start_wav = wav_target.size(-1) - self.segment_size
                    i_start_wav = random.randint(0, max_start_wav)
                else:
                    i_start_wav = 0
                wav_target = wav_target[i_start_wav:i_start_wav + self.segment_size]
            else:
                wav_target = F.pad(wav_target, (0, self.segment_size - wav_target.size(-1)))

            # get mel spectrogram

            spe_target = self.f_melspe(wav_target)
            spe_target = (spe_target - self.mean) / self.std

            self.buffer_dict_target = {"spe": spe_target, "num": num_target}

        return spe_input.unsqueeze(0), num_input, spe_target.unsqueeze(0), num_target

    def __len__(self):
        return self.len_combination
