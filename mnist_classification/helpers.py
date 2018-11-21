from torch import nn


class View(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size

    def forward(self, x):
        b, *_ = x.shape
        return x.view(b, *self.size)
