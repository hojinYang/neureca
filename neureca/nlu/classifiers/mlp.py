import argparse
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule


class MLP(LightningModule):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.input_dim = np.prod(data_config["input_dims"])
        self.output_dim = data_config["output_dims"]
        self.batch_size = data_config["batch_size"]

        self.layer1 = nn.Linear(self.input_dim, 32)
        self.layer2 = nn.Linear(32, self.output_dim)

        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()
        # pl.metics.Clas

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)

        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)

        self.valid_acc(logits.exp(), y)
        self.log("valid_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--load_checkpoint", type=bool)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer