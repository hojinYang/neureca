import argparse
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError
