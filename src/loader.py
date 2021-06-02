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
        train_set, test_set, val_set = self.create_train_test_val()
        batch_size = self.config.loader.batch_size

        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.config.loader.batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=self.config.loader.batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self.config.loader.batch_size, shuffle=True
        )
        self.plot_example_images(self.train_loader)
        print(f"We have {len(self.train_loader) * batch_size} training observations")
        print(f"We have {len(self.val_loader) * batch_size} validation observations")
        print(f"We have {len(self.test_loader) * batch_size} test observations")

    def create_train_test_val(self):
        dataset, targets = self.create_dataset()
        data_index = np.arange(len(dataset))

        train_index, test_index, target_train, target_test = train_test_split(
            data_index,
            targets,
            train_size=self.config.loader.train_size,
            random_state=self.config.loader.random_state,
            shuffle=True,
            stratify=targets,
        )

        train_index, val_index, target_train, target_val = train_test_split(
            train_index,
            target_train,
            train_size=self.config.loader.validation_size,
            random_state=self.config.loader.random_state,
            shuffle=True,
            stratify=target_train,
        )

        train_data = torch.utils.data.Subset(dataset, train_index)
        val_data = torch.utils.data.Subset(dataset, val_index)
        test_data = torch.utils.data.Subset(dataset, test_index)

        return train_data, val_data, test_data

    def create_dataset(self):

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
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
        data = torch.utils.data.ConcatDataset([trainset, testset])
        targets = torch.cat((trainset.targets, testset.targets))

        subsample_data = torch.utils.data.Subset(
            data, range(self.config.loader.subset_size)
        )
        subsample_targets = torch.utils.data.Subset(
            targets, range(self.config.loader.subset_size)
        )
        return subsample_data, subsample_targets

    def plot_example_images(self, loader):
        dataiter = iter(loader)
        image, _ = dataiter.next()
        image = torchvision.utils.make_grid(image)
        np_image = image.numpy()

        fig, axs = plt.subplots(figsize=(15, 10))
        axs.imshow(np.transpose(np_image, (1, 2, 0)))
        axs.axis("off")
        file_name = "../reports/figures/example_images.png"
        fig.savefig(fname=file_name)
        plt.close()
