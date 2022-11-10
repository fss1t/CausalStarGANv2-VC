import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class MelSpectrogramPlotter:
    def __init__(self, h, bar=0):
        self.bar = bar
        self.freq = librosa.mel_frequencies(n_mels=h.num_mels + 1, fmin=h.fmin, fmax=h.fmax)
        self.period_frame = h.hop_size / h.sampling_rate

    def __call__(self, melspe):
        fig, ax = plt.subplots(dpi=150)

        time = np.arange(melspe.shape[1] + 1, dtype=float)
        time = self.period_frame * time
        img = ax.pcolormesh(time, self.freq, melspe, cmap='viridis', norm=Normalize(vmin=-8, vmax=2))

        ax.set_yscale('function', functions=(librosa.hz_to_mel, librosa.mel_to_hz))

        fticks = [250 * 2**n for n in range(0, 5)]
        fticks.insert(0, self.freq[0])
        fticks.append(self.freq[-1])
        ax.set_yticks(fticks)
        tticks = np.arange(*ax.get_xlim(), 0.5)
        ax.set_xticks(tticks)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        if self.bar:
            colorbar = fig.colorbar(img, ax=ax)
            colorbar.set_label("log Amplitude")

        set_size(ax, melspe.shape[1], melspe.shape[0], 4)

        fig.canvas.draw()
        return fig


def set_size(ax, w, h, zoom=1):
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    dpi = ax.figure.get_dpi()

    figw = float(w) / dpi * zoom
    figh = float(h) / dpi * zoom
    figwo, figho = ax.figure.get_size_inches()
    l = l * figwo / figw
    r = 1 - (1 - r) * figwo / figw
    t = 1 - (1 - t) * figho / figh
    b = b * figho / figh
    ax.figure.subplots_adjust(left=l, right=r, top=t, bottom=b)

    figw /= (r - l)
    figh /= (t - b)
    ax.figure.set_size_inches(figw, figh)
