# %% Packages

from utils.args import get_args
from utils.config import process_config
from loader import MNISTDataLoader

# %% Load images


def main():

    args = get_args()
    config = process_config(args.config)

    data = MNISTDataLoader(config)



if __name__ == "__main__":
    main()
