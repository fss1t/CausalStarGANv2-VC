import torch
import torch.utils.data
import numpy as np
from scipy.io.wavfile import read, write
from librosa.util import normalize

MAX_WAV_VALUE = 32768.0


def load_wav(path):
    sampling_rate, data = read(path)
    data = data / MAX_WAV_VALUE
    return data


def write_wav(path, data, sampling_rate):
    data = data.to("cpu")
    data = data * MAX_WAV_VALUE
    data = data.detach().squeeze().numpy().astype(np.int16)
    write(path, sampling_rate, data)
