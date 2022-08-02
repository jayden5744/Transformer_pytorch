import os.path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# EarlyStopping
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int, args: DictConfig, verbose: bool = False, delta: int = 0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.args = args
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_step = 0

    def __call__(self, val_loss, model, epoch, step):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_model(model=model, model_name="best_transformer.pth", epoch=epoch)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("best step :" + str(self.best_step))
                print("best score :" + str(self.val_loss_min))

        else:
            self.best_score = score
            self.save_model(model=model, model_name="best_transformer.pth", epoch=epoch)
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.val_loss_min = val_loss
            self.counter = 0
            self.best_step = step

    def save_model(self, model: nn.Module, model_name: str, epoch: int) -> None:
        model_path = os.path.join(self.args.data.model_path, model_name)
        torch.save(
            {
                'epoch': epoch,
                "data": self.args["data"],
                "trainer": self.args["trainer"],
                "model": self.args["model"],
                'model_state_dict': model.state_dict()
            }, model_path
        )
