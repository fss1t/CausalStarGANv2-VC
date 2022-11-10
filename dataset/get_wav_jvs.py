import os
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

list_set_jvs = ["parallel100",
                "nonpara30"]

list_set_jvsmusic = ["song_common",
                     "song_unique"]


def get_wav_jvs(path_jvs,
                path_jvsmusic,
                path_dir_wav=Path("./wav")):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- get wav from jvs ---")

    # get speaker names

    iters = path_jvs.glob("*")
    speakers = []
    for iter in iters:
        if iter.is_dir():
            speakers.append(iter.name)
    speakers = sorted(speakers)

    # prepare directory

    path_dir_wav.mkdir(exist_ok=1)
    for speaker in speakers:
        path_dir_speaker = path_dir_wav / speaker
        path_dir_speaker.mkdir(exist_ok=1)

    # write wav list

    for speaker in tqdm(speakers):
        for set_ in list_set_jvs:
            path_dir_wav_read = path_jvs / speaker / set_ / "wav24kHz16bit"
            list_path_wav_read_sp = sorted(path_dir_wav_read.glob("*.wav"))

            for path_wav_read in list_path_wav_read_sp:
                path_wav_copy = path_dir_wav / speaker / f"{set_}_{path_wav_read.name}"
                copyfile(path_wav_read, path_wav_copy)

        for set_ in list_set_jvsmusic:
            path_dir_wav_read = path_jvsmusic / speaker / set_ / "wav"
            list_path_wav_read_sp = sorted(path_dir_wav_read.glob("raw.wav"))

            for path_wav_read in list_path_wav_read_sp:
                path_wav_copy = path_dir_wav / speaker / f"{set_}_{path_wav_read.name}"
                copyfile(path_wav_read, path_wav_copy)
