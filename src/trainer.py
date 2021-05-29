# %% Packages

import torch
from tqdm import tqdm

# %% Classes


class MNISTTrainer:
    def __init__(self, data, model, config):
        self.config = config
        self.data = data
        self.model = model

        self.train_model()

    def train_model(self):

        for epoch in tqdm(range(1, self.config.trainer.number_of_epochs + 1)):

            loss_train = 0
            loss_val = 0
            acc_train = 0
            acc_val = 0

            self.model.model.train(True)

            for i, data in enumerate(self.data.trainloader):

                inputs, labels = data
                self.model.optimizer.zero_grad()
                outputs = self.model.model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.model.criterion(outputs, labels)
