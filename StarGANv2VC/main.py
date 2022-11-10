import os
from pathlib import Path
import sys

path_CausalHiFiGAN = "../CausalHiFiGAN"
path_CNNConformer = "../CNNConformer"
path_JDCNet = "../JDCNet"
os.chdir(os.path.dirname(__file__))  # cd .
sys.path.append(path_CausalHiFiGAN)
sys.path.append(path_CNNConformer)
sys.path.append(path_JDCNet)
from StarGANv2VC import make_list, calc_norm, train, infer


def main():
    make_list()
    calc_norm()
    train(path_CNNConformer=path_CNNConformer, path_JDCNet=path_JDCNet)
    # infer()


if __name__ == "__main__":
    main()
