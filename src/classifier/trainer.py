# %% Packages

import matplotlib.pyplot as plt

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

        test_loss, test_accuracy = self.model.model.evaluate(self.data.val_dataset)
        self.plot_history(history, test_loss, test_accuracy)
        print("Test accuracy :", test_accuracy)


    def plot_history(self, history, test_loss, test_accuracy):
        train_acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]

        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]

        fig, axs = plt.subplots(figsize=(10, 5), ncols=2)
        axs = axs.ravel()

        axs[0].plot(train_acc, label="Training Accuracy")
        axs[0].plot(val_acc, label="Validation Accuracy")
        axs[0].scatter(x=(len(train_acc)-1), y=test_accuracy, label="Test Accuracy", marker="X", color="red")
        axs[0].set_title("Accuracy Scores")
        axs[0].legend()

        axs[1].plot(train_loss, label="Training Loss")
        axs[1].plot(val_loss, label="Validation Loss")
        axs[1].scatter(x=(len(train_acc)-1), y=test_loss, label="Test Accuracy", marker="X", color="red")
        axs[1].set_title("Loss Scores")
        axs[1].legend()

        fig.savefig(f"../reports/figures/{self.config.trainer.png_name}.png")

