from pathlib import Path
import json
from attrdict import AttrDict


def load_json(path_json):
    with open(path_json, "r") as js:
        h = json.loads(js.read())
    h = AttrDict(h)
    return h


def load_list(path_list):
    with open(path_list, "r", encoding="utf-8") as txt:
        list_line = txt.read().splitlines()
    list_path = [Path(line) for line in list_line]
    return list_path
