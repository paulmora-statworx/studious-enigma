# %% Packages

import argparse

# %% Functions


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-c",
        "--config",
        dest="config",
        metavar="C",
        default="None",
        help="The configuration file",
    )
    args = argparser.parse_args()
    return args
