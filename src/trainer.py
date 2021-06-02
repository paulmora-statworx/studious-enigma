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

    pass
