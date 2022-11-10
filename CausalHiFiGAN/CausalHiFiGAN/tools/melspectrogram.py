import torch
from librosa.filters import mel as librosa_mel_fn
from functools import partial


class MelSpectrogramExtractor:
    def __init__(self, n_fft, win_size, hop_size, sampling_rate, num_mels, fmin, fmax, padding=True):
        if padding:
            self.pad = (n_fft - hop_size) // 2
        else:
            self.pad = 0

        mel_basis = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        self.mel_basis = torch.from_numpy(mel_basis).float()

        window = torch.hann_window(win_size)
        self.stft = partial(torch.stft,
                            n_fft=n_fft,
                            win_length=win_size,
                            window=window,
                            hop_length=hop_size,
                            center=False,
                            normalized=False)

    def __call__(self, wav):
        if self.pad != 0:
            wav = torch.nn.functional.pad(wav.unsqueeze(-2), (self.pad, self.pad), mode='reflect').squeeze(-2)

        spe = torch.sqrt(torch.sum(self.stft(wav)**2, -1))
        melspe = torch.matmul(self.mel_basis, spe)
        logmelspe = torch.log(torch.clamp(melspe, 1e-5))
        return logmelspe


class MelSpectrogramExtractorForLoss:
    def __init__(self, n_fft, win_size, hop_size, sampling_rate, num_mels, fmin, fmax, padding=True, device="cpu"):
        if padding:
            self.pad = (n_fft - hop_size) // 2
        else:
            self.pad = 0

        mel_basis = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        self.mel_basis = torch.from_numpy(mel_basis).float().to(device)

        window = torch.hann_window(win_size).to(device)
        self.stft = partial(torch.stft,
                            n_fft=n_fft,
                            win_length=win_size,
                            window=window,
                            hop_length=hop_size,
                            center=False,
                            normalized=False)

    def __call__(self, wav):
        if self.pad != 0:
            wav = torch.nn.functional.pad(wav.unsqueeze(-2), (self.pad, self.pad), mode='reflect').squeeze(-2)

        spe = torch.sqrt(torch.sum(self.stft(wav)**2, -1) + 1e-9)
        melspe = torch.matmul(self.mel_basis, spe)
        logmelspe = torch.log(torch.clamp(melspe, 1e-5))
        return logmelspe
