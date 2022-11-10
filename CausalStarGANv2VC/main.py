import os
from pathlib import Path
import sys

path_CausalHiFiGAN = "../CausalHiFiGAN"
path_CNNConformer = "../CNNConformer"
path_JDCNet = "../JDCNet"
path_StarGANv2VC = "../StarGANv2VC"
os.chdir(os.path.dirname(__file__))  # cd .
sys.path.append(path_CausalHiFiGAN)
sys.path.append(path_CNNConformer)
sys.path.append(path_JDCNet)
sys.path.append(path_StarGANv2VC)
from CausalStarGANv2VC import make_list, calc_norm, train, infer


def main():
    make_list()
    calc_norm()
    train(path_CNNConformer=path_CNNConformer, path_JDCNet=path_JDCNet, path_StarGANv2VC=path_StarGANv2VC)
    infer(path_CausalHiFiGAN=path_CausalHiFiGAN)


if __name__ == "__main__":
    main()
