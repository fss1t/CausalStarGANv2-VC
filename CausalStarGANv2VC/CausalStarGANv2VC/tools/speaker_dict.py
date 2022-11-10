

def get_dict_speaker(path_dir_list):
    list_path_list = sorted(path_dir_list.glob("*.txt"))
    list_speaker = [path_list.stem for path_list in list_path_list]
    return list_speaker


def inverse_dict_speaker(list_speaker):
    dict_num_speaker = {name: num for num, name in enumerate(list_speaker)}
    return dict_num_speaker
