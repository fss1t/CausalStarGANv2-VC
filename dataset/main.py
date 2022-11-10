from argparse import ArgumentParser
from get_wav_jvs import get_wav_jvs
from get_lab_jvs import get_lab_jvs


def main():
    parser = ArgumentParser()
    parser.add_argument("--path_jvs", default="../jvs_ver1")
    parser.add_argument("--path_jvs_music", default="../jvs_music_ver1")
    a = parser.parse_args()
    get_wav_jvs(a.path_jvs, a.path_jvs_music)
    get_lab_jvs(a.path_jvs)


if __name__ == "__main__":
    main()
