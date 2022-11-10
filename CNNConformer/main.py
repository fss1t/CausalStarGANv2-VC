import os
from pathlib import Path
import sys

path_CausalHiFiGAN = "../CausalHiFiGAN"
os.chdir(os.path.dirname(__file__))  # cd .
sys.path.append(path_CausalHiFiGAN)
from CNNConformer import make_list, make_dict_phoneme, calc_norm, train


def main():
    make_list()
    make_dict_phoneme()
    calc_norm()
    train()


if __name__ == "__main__":
    main()
