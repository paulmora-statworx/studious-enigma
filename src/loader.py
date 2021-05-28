# %% Packages

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

# %% Classes


class MNISTDataLoader:
    def __init__(self, config):
        self.config = config
        self.create_loaders()

    def create_loaders(self):
        trainset, testset, valset = self.create_train_test_val()

        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.config.loader.batch_size, shuffle=True
        )
        self.valloader = torch.utils.data.DataLoader(
            valset, batch_size=self.config.loader.batch_size, shuffle=True
        )
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.config.loader.batch_size, shuffle=True
        )
        batch_size = self.config.loader.batch_size
        print(
            f"We have {len(self.trainloader) * batch_size} training observations"
        )
        print(
            f"We have {len(self.valloader) * batch_size} validation observations"
        )
        print(
            f"We have {len(self.testloader) * batch_size} test observations"
        )

    def create_train_test_val(self):
        dataset, targets = self.create_dataset()

        trainset, testset, target_train, target_test = train_test_split(
            dataset,
            targets,
            train_size=self.config.loader.train_size,
            random_state=self.config.loader.random_state,
            shuffle=True,
            stratify=targets,
        )

        trainset, valset, target_train, target_val = train_test_split(
            trainset,
            target_train,
            train_size=self.config.loader.validation_size,
            random_state=self.config.loader.random_state,
            shuffle=True,
            stratify=target_train,
        )

        self.plot_distribution(target_train, target_test, target_val)
        return trainset, testset, valset

    def create_dataset(self):

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        )
        trainset = torchvision.datasets.MNIST(
            root=self.config.loader.root,
            train=True,
            download=False,
            transform=transform,
        )
        testset = torchvision.datasets.MNIST(
            root=self.config.loader.root,
            train=False,
            download=False,
            transform=transform,
        )
        dataset = ConcatDataset([trainset, testset])
        targets = torch.cat((trainset.targets, testset.targets))
        return dataset, targets

    def plot_example_images(self, loader):
        dataiter = iter(loader)
        image, _ = dataiter.next()
        image = torchvision.utils.make_grid(image)

        image = image / 2 + 0.5
        np_image = image.numpy()

        fig, axs = plt.subplots(figsize=(15, 10))
        axs.imshow(np.transpose(np_image, (1, 2, 0)))
        axs.axis("off")
        file_name = "../reports/figures/example_images.png"
        fig.savefig(fname=file_name)
        plt.close()
