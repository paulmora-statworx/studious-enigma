# %% Packages

import json
from dotmap import DotMap

# %% Functions


def get_config_from_json(json_file):

    with open(json_file, "r") as config_file:
        config_dict = json.load(config_file)
    config = DotMap(config_dict)
    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    return config
