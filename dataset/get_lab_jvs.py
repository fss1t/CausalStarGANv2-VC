import os
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

list_set_jvs = ["parallel100",
                "nonpara30"]

list_datalacker = ["jvs006", "jvs028"]


def get_lab_jvs(path_jvs,
                path_dir_lab=Path("./lab")):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- get wav from jvs ---")

    # get speaker names

    iters = path_jvs.glob("*")
    speakers = []
    for iter in iters:
        if iter.is_dir():
            speakers.append(iter.name)
    speakers = sorted(speakers)
    cut_datalacked_speaker(speakers)

    # prepare directory

    path_dir_lab.mkdir(exist_ok=1)
    for speaker in speakers:
        path_dir_speaker = path_dir_lab / speaker
        path_dir_speaker.mkdir(exist_ok=1)

    # write lab list

    for speaker in tqdm(speakers):
        for set_ in list_set_jvs:
            path_dir_lab_read = path_jvs / speaker / set_ / "lab/mon"
            list_path_lab_read_sp = sorted(path_dir_lab_read.glob("*.lab"))

            for path_lab_read in list_path_lab_read_sp:
                path_lab_copy = path_dir_lab / speaker / f"{set_}_{path_lab_read.name}"
                copyfile(path_lab_read, path_lab_copy)


def cut_datalacked_speaker(speakers):
    for lacker in list_datalacker:
        speakers.remove(lacker)
