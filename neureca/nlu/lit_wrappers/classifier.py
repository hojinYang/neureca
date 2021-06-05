import argparse
from torch.nn import functional as F
import torchmetrics
from neureca.shared.lit_wrapper import BaseLitWrapper


class Classifier(BaseLitWrapper):
    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x):
        self.model(x)

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
        self.log("val_acc", self.valid_acc, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.test_acc(logits.exp(), y)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)

    def get_main_validation_metric(self):
        return {"name": "val_acc", "mode": "max"}

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--load_checkpoint", type=bool)
        return parser