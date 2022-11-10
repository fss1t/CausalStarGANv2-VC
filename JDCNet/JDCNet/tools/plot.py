import math as m
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator


class F0Plotter:
    def __init__(self, h, hv, range_octave=(-1, 2), labels=["target", "predicted"]):
        self.labels = labels

        range_label = range(int(np.ceil(range_octave[0])), int(np.floor(range_octave[1]) + 1))
        self.fticks = [n for n in range_label]
        self.fticklabels = [str(int(h.f0_base * 2**n)) for n in range_label]

        self.period_frame = hv.hop_size / hv.sampling_rate

    def __call__(self, *list_f0):
        fig, ax = plt.subplots(dpi=150)

        for f0 in list_f0:
            time = np.arange(f0.shape[0], dtype=float)
            time = self.period_frame * time
            ax.plot(time, f0, lw=1)

        ax.set_yticks(self.fticks)
        ax.set_yticklabels(self.fticklabels)
        ax.yaxis.set_minor_locator(MultipleLocator(1 / 12))
        tticks = np.arange(0, self.period_frame * f0.shape[0], 0.5)
        ax.set_xticks(tticks)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.grid(which="major", axis="y", linewidth=0.5, color="gray")
        ax.grid(which="minor", axis="y", linewidth=0.5, color="lightgray")

        ax.legend(self.labels)
        fig.set_size_inches(9.6, 4.8)

        fig.canvas.draw()
        return fig
