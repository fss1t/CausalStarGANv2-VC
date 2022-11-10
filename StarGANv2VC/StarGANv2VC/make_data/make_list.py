from pathlib import Path, PurePosixPath
from tqdm import tqdm

dict_set_jvs = {
    "train": {"parallel100": range(0, 95),
              "nonpara30": None},
    "valid": {"parallel100": range(-5, 0)}}

dict_set_jvsmusic = {
    "train": {"song_common": None,
              "song_unique": None}
}

dict_target_valid = {
    "jvs004": {"parallel100": [13]},
    "jvs010": {"parallel100": [13]}}

dict_speaker_sample = {
    "jvs068": {"parallel100": [-5]}}


weight_batching_singing = 10


def make_list(path_dataset=Path("../dataset"),
              path_dir_list=Path("./data/list")):
    print("--- make wav file list ---")

    # prepare directory

    path_dataset_wav = path_dataset / "wav"
    path_dir_list.mkdir(exist_ok=1, parents=1)
    for use in dict_set_jvs.keys():
        (path_dir_list / use).mkdir(exist_ok=1)
    (path_dir_list / "valid_target").mkdir(exist_ok=1)

    # get speaker names

    iters = path_dataset_wav.glob("*")
    speakers = []
    for iter in iters:
        if iter.is_dir():
            speakers.append(iter.name)
    speakers = sorted(speakers)

    # make wav path list

    for speaker in tqdm(speakers):
        txt = {}
        for use in dict_set_jvs.keys():
            txt[use] = open(path_dir_list / use / f"{speaker}.txt", "w", encoding="utf-8")

        for use, dict_set in dict_set_jvs.items():
            for set_, numbers in dict_set.items():
                path_dir_wav = path_dataset_wav / speaker
                list_path_wav_sp = sorted(path_dir_wav.glob(f"{set_}_*.wav"))
                if numbers is not None:
                    list_path_wav_sp = [list_path_wav_sp[i] for i in numbers]

                for path_wav in list_path_wav_sp:
                    txt[use].write(str(PurePosixPath(path_wav.resolve())) + "\n")

        for use, dict_set in dict_set_jvsmusic.items():
            for set_, numbers in dict_set.items():
                path_dir_wav = path_dataset_wav / speaker
                list_path_wav_sp = sorted(path_dir_wav.glob(f"{set_}_*.wav"))

                for path_wav in list_path_wav_sp:
                    if use == "train":
                        for _ in range(weight_batching_singing):
                            txt[use].write(str(PurePosixPath(path_wav.resolve())) + "\n")
                    else:
                        txt[use].write(str(PurePosixPath(path_wav.resolve())) + "\n")

        for use in dict_set_jvs.keys():
            txt[use].close()

    # make target wav path list

    txt_target = {}
    for speaker, dict_set in dict_target_valid.items():
        txt_target = open(path_dir_list / "valid_target" / f"{speaker}.txt", "w", encoding="utf-8")
        for set_, numbers in dict_set.items():
            path_dir_wav = path_dataset_wav / speaker
            list_path_wav_sp = sorted(path_dir_wav.glob(f"{set_}_*.wav"))
            for i in numbers:
                txt_target.write(str(PurePosixPath(list_path_wav_sp[i].resolve())) + "\n")
        txt_target.close()

    # make wav path list for visualize

    txt_sample = open(path_dir_list / "valid_sample.txt", "w", encoding="utf-8")
    for speaker, dict_set in dict_speaker_sample.items():
        for set_, numbers in dict_set.items():
            path_dir_wav = path_dataset_wav / speaker
            list_path_wav_sp = sorted(path_dir_wav.glob(f"{set_}_*.wav"))
            for i in numbers:
                txt_sample.write(str(PurePosixPath(list_path_wav_sp[i].resolve())) + "\n")
    txt_sample.close()
