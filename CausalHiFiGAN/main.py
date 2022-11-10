import os
from CausalHiFiGAN import make_list, train, infer


def main():
    os.chdir(os.path.dirname(__file__))  # cd .

    make_list()
    train()
    infer()


if __name__ == "__main__":
    main()
