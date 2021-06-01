import argparse
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__()

        self.input_dims = data_config["input_dims"]
        self.output_dims = data_config["output_dims"]  # OUTPUT C

        self.layer1 = nn.Linear(self.input_dim, 32)
        self.layer2 = nn.Linear(32, self.output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)

        x = F.log_softmax(x, dim=1)
        return x
