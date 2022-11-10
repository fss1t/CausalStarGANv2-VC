from pathlib import Path, PurePosixPath
from tqdm import tqdm


dict_set_jvs = {
    "train": {"parallel100": range(0, 95),
              "nonpara30": None},
    "valid": {"parallel100": range(-5, 0)}}

dict_speaker_sample = {
    "jvs068": {"parallel100": [-5]},
    "jvs004": {"parallel100": [-5]},
    "jvs010": {"parallel100": [-5]}}


def make_list(path_dataset=Path("../dataset"),
              path_dir_list=Path("./data/list")):
    print("--- make wav file list ---")

    # prepare directory

    path_dir_wav = path_dataset / "wav"
    path_dir_lab = path_dataset / "lab"
    path_dir_list.mkdir(exist_ok=1, parents=1)

    # get speaker names

    iters = path_dir_lab.glob("*")
    speakers = []
    for iter in iters:
        if iter.is_dir():
            speakers.append(iter.name)
    speakers = sorted(speakers)

    # make wav path list file

    txt_wav, txt_lab = {}, {}
    for use in dict_set_jvs.keys():
        txt_wav[use] = open(path_dir_list / f"wav_{use}.txt", "w", encoding="utf-8")
        txt_lab[use] = open(path_dir_list / f"lab_{use}.txt", "w", encoding="utf-8")

    # write wav path

    for speaker in tqdm(speakers):
        for use, dict_set in dict_set_jvs.items():
            for set_, numbers in dict_set.items():
                path_dir_wav_sp = path_dir_wav / speaker
                list_path_wav_sp = sorted(path_dir_wav_sp.glob(f"{set_}_*.wav"))
                if numbers is not None:
                    list_path_wav_sp = [list_path_wav_sp[i] for i in numbers]

                for path_wav in list_path_wav_sp:
                    path_lab = path_dir_lab / path_wav.parent.stem / f"{path_wav.stem}.lab"
                    if path_lab.exists():
                        txt_wav[use].write(str(PurePosixPath(path_wav.resolve())) + "\n")
                        txt_lab[use].write(str(PurePosixPath(path_lab.resolve())) + "\n")

    # close file pointer

    for use in dict_set_jvs.keys():
        txt_wav[use].close()
        txt_lab[use].close()

    # make wav path list for visualize

    txt_sample = open(path_dir_list / "valid_sample.txt", "w", encoding="utf-8")
    for speaker, dict_set in dict_speaker_sample.items():
        for set_, numbers in dict_set.items():
            path_dir_wav_sp = path_dir_wav / speaker
            list_path_wav_sp = sorted(path_dir_wav_sp.glob(f"{set_}_*.wav"))
            for i in numbers:
                txt_sample.write(str(PurePosixPath(list_path_wav_sp[i].resolve())) + "\n")
    txt_sample.close()
