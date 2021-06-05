import argparse
import torch
import torchmetrics
from torchcrf import CRF
import numpy as np
from neureca.shared.lit_wrapper import BaseLitWrapper

# (B, C, S) logits, where S is the length of the sequence and C is the number of classes


class CRFRecognizer(BaseLitWrapper):
    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        self.crf = CRF(self.model.output_dims, batch_first=True)
        self.mask_index = self.model.output_dims
        num_classes = self.model.output_dims + 1

        self.valid_f1 = torchmetrics.F1(
            num_classes=num_classes, average="macro", ignore_index=self.mask_index
        )
        self.test_f1 = torchmetrics.F1(
            num_classes=num_classes, average="macro", ignore_index=self.mask_index
        )

        self.valid_prec = torchmetrics.Precision(
            num_classes=num_classes, average="macro", ignore_index=self.mask_index
        )
        self.test_prec = torchmetrics.Precision(
            num_classes=num_classes, average="macro", ignore_index=self.mask_index
        )

        self.valid_recall = torchmetrics.Recall(
            num_classes=num_classes, average="macro", ignore_index=self.mask_index
        )
        self.test_recall = torchmetrics.Recall(
            num_classes=num_classes, average="macro", ignore_index=self.mask_index
        )

    def forward(self, x):
        self.model(x)

    def training_step(self, batch, batch_index):
        x, y, mask = batch
        # x -> (B, S, E)
        # y -> (B, S)
        # mask -> (B, S)
        logits = self(x)  # -> (B, S, C)
        loss = -self.crf(logits, y, mask=mask)

        return loss

    def decode(self, x, mask):
        logits = self(x)
        return self.crf.decode(logits, mask=mask)

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        logits = self(x)  # -> (B, S, C)

        loss = -self.crf(logits, y, mask=mask)
        self.log("val_loss", loss, prog_bar=True)

        preds_ = self.crf.decode(logits, mask=mask)
        preds = np.full_like(y, self.mask_index)

        for i, j in enumerate(preds_):
            preds[i][: len(j)] = j

        preds = torch.tensor(preds).view(-1)
        y[mask == 0] = self.mask_index
        y = y.view(-1)

        self.valid_f1(preds, y)
        self.valid_prec(preds, y)
        self.valid_recall(preds, y)

        self.log("valid_f1", self.valid_f1, on_epoch=True, prog_bar=True)
        self.log("valid_prec", self.valid_prec, on_epoch=True, prog_bar=True)
        self.log("valid_recall", self.valid_recall, on_epoch=True, prog_bar=True)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--load_checkpoint", type=bool)

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer