import torch

from CausalHiFiGAN.tools.file_io import load_json
from .ctc import CTC


class CTCDecoder:
    def __init__(self, path_phoneme, blank=-1):
        self.ctc = CTC(blank)

        dict_phoneme = load_json(path_phoneme)
        self.inverse_dict = {code: phoneme for phoneme, code in dict_phoneme.items()}

    def __call__(self, prob):
        lab = self.ctc(prob)
        sentence = [self.inverse_dict[code] for code in lab.tolist()]
        sentence = " ".join(sentence)
        return sentence
