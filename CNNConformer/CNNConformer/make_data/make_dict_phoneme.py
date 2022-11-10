from pathlib import Path
import json
from CausalHiFiGAN.tools.file_io import load_list


def make_dict_phoneme(path_dir_list=Path("./data/list"),
                      path_dir_param=Path("./data/param")):
    print("--- make dict phoneme to code ---")

    # prepare directory

    path_dir_param.mkdir(exist_ok=1, parents=1)

    # load lab

    list_path_lab = load_list(path_dir_list / "lab_train.txt")
    list_path_lab.extend(load_list(path_dir_list / "lab_valid.txt"))

    set_phoneme = set()
    for path_lab in list_path_lab:
        with open(path_lab, "r") as txt:
            for line in txt.read().splitlines():
                set_phoneme.add(line.rstrip().split(" ")[2])

    # make phoneme dict

    phonemes_sil = {"sil", "pau"}
    set_phoneme = set_phoneme - phonemes_sil

    dict_phoneme = {}
    for phoneme_sil in phonemes_sil:
        dict_phoneme[phoneme_sil] = 0
    for i, phoneme in enumerate(sorted(set_phoneme), 1):
        dict_phoneme[phoneme] = i

    with open(path_dir_param / "phoneme.json", "w") as js:
        json.dump(dict_phoneme, js, indent=4)

    print(dict_phoneme)
