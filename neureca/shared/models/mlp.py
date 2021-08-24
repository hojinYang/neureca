import argparse
import torch.nn as nn
from torch.nn import functional as F
from .base_model import BaseModel

MLP_DIMS = [128, 64]
MLP_DROPOUT = 0.2


class MLP(BaseModel):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__(data_config, args)
        mlp_dims = self.args.get("mlp_dims", MLP_DIMS)
        mlp_dropout = self.args.get("mlp_dropout", MLP_DROPOUT)

        if isinstance(mlp_dims, int):
            mlp_dims = [mlp_dims]
        mlp_dims = [self.input_dims] + mlp_dims + [self.output_dims]

        self.layers = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(mlp_dims[:-1], mlp_dims[1:])):
            self.layers.append(nn.Dropout(mlp_dropout))
            self.layers.append(nn.Linear(d_in, d_out))
            if i != len(mlp_dims) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        for layer in self.layers:
            x = layer(x)
        x = F.log_softmax(x, dim=1)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--mlp_dims", type=int, nargs="+", default=MLP_DIMS)
        parser.add_argument("--mlp_dropout", type=int, default=MLP_DROPOUT)

        return parser
