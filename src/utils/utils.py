import numpy as np
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# EarlyStopping
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
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
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_step = 0

    def __call__(self, val_loss, model, step, encoder_parameter=None, decoder_parameter=None,
                 seq_len=50):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, encoder_parameter, decoder_parameter, seq_len)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("best step :" + str(self.best_step))
                print("best score :" + str(self.val_loss_min))
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, encoder_parameter, decoder_parameter, seq_len)
            self.counter = 0
            self.best_step = step

    def save_checkpoint(self, val_loss, model, encoder_parameter, decoder_parameter, seq_len):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'seq_len': seq_len,
            'encoder_parameter': encoder_parameter,
            'decoder_parameter': decoder_parameter,
            'model_state_dict': model.state_dict()
        }, 'Model/best_transformer.pth')
        self.val_loss_min = val_loss