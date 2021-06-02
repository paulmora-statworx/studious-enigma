# %% Packages

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

# %% Classes


class MNISTDataLoader:
    def __init__(self, config):
        self.config = config
        self.train_dataset, self.val_dataset, self.test_dataset = self.create_dataset()

        number_train_images = len(self.train_dataset) * self.config.loader.batch_size
        number_val_images = len(self.val_dataset) * self.config.loader.batch_size
        number_test_images = len(self.test_dataset) * self.config.loader.batch_size

        print(f"Number of training images: {number_train_images}")
        print(f"Number of validation images: {number_val_images}")
        print(f"Number of testing images: {number_test_images}")

    def create_dataset(self):
        dataset = self.load_dataset()
        dataset_size = len(dataset)
        test_val_size = (1 - self.config.loader.train_size) / 2

        train_size = int(self.config.loader.train_size * dataset_size)
        val_size = int(test_val_size * dataset_size)
        test_size = int(test_val_size * dataset_size)

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        val_dataset = test_dataset.skip(val_size)
        test_dataset = test_dataset.take(test_size)

        return train_dataset, val_dataset, test_dataset

    def load_dataset(self):
        image_size = (self.config.loader.target_size, self.config.loader.target_size)
        dataset = image_dataset_from_directory(
            self.config.loader.root,
            batch_size=self.config.loader.batch_size,
            shuffle=True,
            image_size=image_size,
        )
        self.plot_images(dataset)
        return dataset

    def plot_images(self, dataset):
        images, _ = next(iter(dataset))
        fig, axs = plt.subplots(figsize=(20, 10), ncols=self.config.loader.examples)
        axs = axs.ravel()
        for i, image in enumerate(images[: self.config.loader.examples]):
            np_image = image.numpy().astype(np.uint8)
            axs[i].imshow(np_image)
            axs[i].axis("off")
        fig.savefig(fname="../reports/figures/example_images.png", bbox_inches="tight")
        plt.close()
