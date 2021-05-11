import argparse
import torch
from torch import nn
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule
from torchcrf import CRF
import numpy as np

LSTM_DIM = 512
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2

# (B, C, S) logits, where S is the length of the sequence and C is the number of classes


class LSTMCRF(LightningModule):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__()

        self.args = vars(args) if args is not None else {}

        lstm_dim = self.args.get("lstm_dim", LSTM_DIM)
        lstm_layers = self.args.get("lstm_layers", LSTM_LAYERS)
        lstm_dropout = self.args.get("lstm_dropout", LSTM_DROPOUT)

        self.input_dims = data_config[
            "input_dims"
        ]  # (S, E), where S is the length of the sequence and H is the hidden dim

        self.output_dims = data_config["output_dims"]  # OUTPUT C
        self.ignore_index = self.output_dims

        self.batch_size = data_config["batch_size"]

        self.lstm = nn.LSTM(
            input_size=self.input_dims[1],
            hidden_size=lstm_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_dim, self.output_dims)
        self.crf = CRF(self.output_dims, batch_first=True)

        self.valid_f1 = torchmetrics.F1(
            num_classes=self.output_dims + 1, average="macro", ignore_index=self.ignore_index
        )
        self.test_f1 = torchmetrics.F1(
            num_classes=self.output_dims + 1, average="macro", ignore_index=self.ignore_index
        )

        self.valid_prec = torchmetrics.Precision(
            num_classes=self.output_dims + 1, average="macro", ignore_index=self.ignore_index
        )
        self.test_prec = torchmetrics.Precision(
            num_classes=self.output_dims + 1, average="macro", ignore_index=self.ignore_index
        )

        self.valid_recall = torchmetrics.Recall(
            num_classes=self.output_dims + 1, average="macro", ignore_index=self.ignore_index
        )
        self.test_recall = torchmetrics.Recall(
            num_classes=self.output_dims + 1, average="macro", ignore_index=self.ignore_index
        )

    def forward(self, x):

        # x -> (B, S, E)
        B, S, _E = x.shape
        x = x.permute(1, 0, 2)  # -> (S, B, E)

        x, _ = self.lstm(x)  # -> (S, B, 2 * H) where H is lstm_dim

        # Sum up both directions of the LSTM:
        x = x.view(S, B, 2, -1).sum(dim=2)  # -> (S, B, H)

        x = self.fc(x)  # -> (S, B, C)

        return x.permute(1, 0, 2)  # -> (B, S, C)

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
        preds = np.full_like(y, self.ignore_index)

        for i, j in enumerate(preds_):
            preds[i][: len(j)] = j

        preds = torch.tensor(preds)
        preds = preds.view(-1)

        y[mask == 0] = self.ignore_index
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
        parser.add_argument("--lstm_dim", type=int, default=LSTM_DIM)
        parser.add_argument("--lstm_layers", type=int, default=LSTM_LAYERS)

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer