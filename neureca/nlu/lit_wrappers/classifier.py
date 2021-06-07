import argparse
import torch
from torch.nn import functional as F
import torchmetrics
from neureca.shared.lit_wrapper import BaseLitWrapper


OPTIMIZER = "Adam"
LR = 1e-3


class Classifier(BaseLitWrapper):
    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get("lr", LR)

        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

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

        self.val_acc(logits.exp(), y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.test_acc(logits.exp(), y)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def get_main_validation_metric(self):
        return {"name": "val_acc", "mode": "max"}

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--load_checkpoint", type=bool)
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER)
        parser.add_argument("--lr", type=float, default=LR)
        return parser