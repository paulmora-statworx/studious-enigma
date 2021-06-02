# %% Packages

from torchvision import models
import torch.optim as optim
from torch.nn import Sequential, Linear, CrossEntropyLoss
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# %% Classes


class MNISTModelLoader:
    def __init__(self, data, config):
        self.data = data
        self.config = config

    pass
