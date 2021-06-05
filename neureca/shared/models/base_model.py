import argparse
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.input_dims = data_config["input_dims"]
        self.output_dims = data_config["output_dims"]

    def forward(self, x):
        raise NotImplementedError
