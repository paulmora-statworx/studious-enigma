# %% Packages


# %% Classes


class MNISTTrainer:
    def __init__(self, data, model, config):
        self.config = config
        self.data = data
        self.model = model

        self.train_model()

    def train_model(self):
        history = self.model.model.fit(
            self.data.train_dataset,
            epochs=self.config.trainer.number_of_epochs,
            validation_data=self.data.val_dataset,
        )
