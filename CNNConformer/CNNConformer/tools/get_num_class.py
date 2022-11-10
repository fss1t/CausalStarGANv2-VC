import json


def get_num_class(path_phoneme):
    with open(path_phoneme) as js:
        dict_phoneme = json.loads(js.read())
    inverse_dict = {code: phoneme for phoneme, code in dict_phoneme.items()}
    return len(inverse_dict)
