# %% Packages

from utils.args import get_args
from utils.config import process_config
from generator import generator_model
from discriminator import discriminator_model
from trainer import GanTrainer

# %% Load images


def main():

    args = get_args()
    config = process_config(args.config)

    print("Calling Generator Model")
    generator = generator_model(config)

    print("Calling Discriminator Model")
    discriminator = discriminator_model(config)

    print("Training the model")
    GanTrainer(generator, discriminator, config)

if __name__ == "__main__":
    main()
