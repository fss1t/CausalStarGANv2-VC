import math as m
import numpy as np
import pyworld
import pyreaper

MAX_WAV_VALUE = 32768.0


class F0Extractor:
    def __init__(self, hop_size, sampling_rate, f0_floor, f0_ceil, f0_base=220):
        self.sampling_rate = sampling_rate
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.frame_period = hop_size / sampling_rate / 2
        self.lf0_base = m.log2(f0_base)

    def __call__(self, wav):
        wav_int = (wav * MAX_WAV_VALUE).astype(np.int16)
        _, _, _, f0_mask, _ = pyreaper.reaper(wav_int, self.sampling_rate, frame_period=self.frame_period)

        wav = wav.astype(np.float64)
        f0, _ = pyworld.harvest(wav,
                                self.sampling_rate,
                                f0_floor=self.f0_floor,
                                f0_ceil=self.f0_ceil,
                                frame_period=self.frame_period * 1000)

        f0_mask = np.pad(f0_mask, (1, len(f0) - len(f0_mask) - 1))
        f0 = np.where(f0_mask == -1.0, 0.0, f0)

        f0 = f0[1::2]
        log2f0 = np.where(f0 == 0.0, 0.0, np.log2(f0) - self.lf0_base)
        vuv = np.where(f0 == 0.0, 0.0, 1.0)
        return log2f0, vuv
