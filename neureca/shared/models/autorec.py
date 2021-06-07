import argparse
import torch
import torch.nn as nn
from .base_model import BaseModel

AE_HIDDEN_DIM = 64
AE_ZEROOUT = 0.5


class AutoRec(BaseModel):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__(data_config, args)
        if self.input_dims != self.output_dims:
            raise ValueError("Input dims and output dims should be equal")
        hidden_dim = self.args.get("ae_hidden_dim", AE_HIDDEN_DIM)
        zeroout = self.args.get("ae_zeroout", AE_ZEROOUT)

        self.enc = nn.Linear(self.input_dims, hidden_dim)
        self.dec = nn.Linear(hidden_dim, self.output_dims)
        self.zeroout = nn.Dropout(zeroout)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.zeroout(x)
        x = x / (x.sum(dim=1, keepdim=True) + 1e-10)

        x = self.enc(x)
        x = torch.tanh(x)
        x = self.dec(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--ae_hidden_dim", type=int, default=AE_HIDDEN_DIM)
        parser.add_argument("--ae_zeroout", type=float, default=AE_ZEROOUT)

        return parser
