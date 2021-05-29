# %% Packages

from utils.args import get_args
from utils.config import process_config
from loader import MNISTDataLoader
from model import MNISTModelLoader
from trainer import MNISTTrainer

# %% Load images


def main():

    args = get_args()
    config = process_config(args.config)

    print("Loading the data")
    data = MNISTDataLoader(config)

    print("Calling the model")
    model = MNISTModelLoader(data, config)

    print("Training the model")
    trainer = MNISTTrainer(data, model, config)


if __name__ == "__main__":
    main()
