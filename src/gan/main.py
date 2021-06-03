# %% Packages

from utils.args import get_args
from utils.config import process_config
from generator import generator_model

# %% Load images


def main():

    args = get_args()
    config = process_config(args.config)

    print("Calling Generator Model")
    generator = generator_model(config)

    print("Calling Discriminator Model")

    print("Training the model")


if __name__ == "__main__":
    main()
