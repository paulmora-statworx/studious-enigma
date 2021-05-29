# %% Packages

from torchvision import models
import torch.optim as optim
from torch.nn import Sequential, Linear, CrossEntropyLoss

# %% Classes


class MNISTModelLoader:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.load_model()

    def load_model(self):
        self.model = models.vgg16_bn(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier[6] = Sequential(Linear(4096, 10))
        for param in self.model.classifier[6].parameters():
            param.requires_grad = True

        self.criterion = CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.classifier[6].parameters(), lr=self.config.model.lr
        )
