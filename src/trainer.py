# %% Packages

import torch
from tqdm import tqdm
from torch.autograd import Variable

# %% Classes


class MNISTTrainer:
    def __init__(self, data, model, config):
        self.config = config
        self.data = data
        self.model = model

        self.train_model()

    def train_model(self):

        loss_train_list = []
        loss_val_list = []
        acc_train_list = []
        acc_val_list = []

        for epoch in tqdm(range(1, self.config.trainer.number_of_epochs + 1)):

            loss_train = 0
            loss_val = 0
            number_of_hits_train = 0
            number_of_hits_val = 0

            self.model.model.train(True)

            number_of_train_batches = len(self.data.train_loader)
            number_of_val_batches = len(self.data.val_loader)

            # Training
            for i, data in enumerate(self.data.train_loader):

                if i % 10 == 0:
                    print(f"Training batch {i}/{number_of_train_batches}")

                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                self.model.optimizer.zero_grad()
                outputs = self.model.model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.model.criterion(outputs, labels)

                loss.backward()
                self.model.optimizer.step()

                loss_value = loss.data.tolist()
                hit_value = (sum(preds == labels.data)).tolist()

                loss_train += loss_value
                number_of_hits_train += hit_value

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            self.model.model.train(False)
            self.model.model.eval()

            # Validation
            for i, data in enumerate(self.data.val_loader):

                if i % 10 == 0:
                    print(f"Validation batch {i}/{number_of_val_batches}")

                inputs, labels = data
                inputs, labels = Variable(inputs, volatile=True), Variable(
                    labels, volatile=True
                )

                self.model.optimizer.zero_grad()
                outputs = self.model.model(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = self.model.criterion(outputs, labels)

                loss.backward()
                self.model.optimizer.step()

                loss_value = loss.data.tolist()
                hit_value = (sum(preds == labels.data)).tolist()

                loss_val += loss_value
                number_of_hits_val += hit_value

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            # Calculate figures
            avg_loss_train = loss_train / len(self.data.train_loader)
            avg_loss_val = loss_val / len(self.data.val_loader)
            accuracy_train = number_of_hits_train / len(self.data.train_loader)
            accuracy_val = number_of_hits_val / len(self.data.val_loader)

            # Appending data
            loss_train_list.append(avg_loss_train)
            loss_val_list.append(avg_loss_val)
            acc_train_list.append(accuracy_train)
            acc_val_list.append(accuracy_val)

            # Printing
            print(f"Epoch Number: {epoch}")
            print(f"Average loss train: {avg_loss_train}")
            print(f"Average loss validation: {avg_loss_val}")
            print(f"Accuracy train: {accuracy_train}")
            print(f"Accuracy validation: {accuracy_val}")
