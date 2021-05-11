import argparse
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule


class AE(LightningModule):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        self.input_dim = data_config["input_dims"]
        self.output_dim = self.input_dim

        self.enc = nn.Linear(self.input_dim, 32)
        self.dec = nn.Linear(32, self.output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        x = self.enc(x)
        x = F.sigmoid(x)
        x = self.dec(x)
        return x

    def training_step(self, batch, batch_index):
        x, y = batch
        x[x > 0] = 1.0
        y[y > 0] = 1.0
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(1).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = x + y

        x[x > 0] = 1.0
        y[y > 0] = 1.0

        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(1).mean()
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = x + y
        x[x > 0] = 1.0
        y[y > 0] = 1.0

        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(1).mean()
        self.log("test_loss", loss, prog_bar=True)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--load_checkpoint", type=bool)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer