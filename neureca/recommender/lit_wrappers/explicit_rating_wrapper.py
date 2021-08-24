import argparse
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity
from neureca.shared.lit_wrapper import BaseLitWrapper

OPTIMIZER = "Adam"
LR = 1e-3


class ExplicitRatingWrapper(BaseLitWrapper):
    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__(model, args)

        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get("lr", LR)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_index):
        x, y = batch

        preds = self(x)
        loss = F.mse_loss(preds, y, reduction="none").sum(1).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = x + y

        preds = self(x)
        loss = F.mse_loss(preds, y, reduction="none").sum(1).mean()
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = x + y

        preds = self(x)
        loss = F.mse_loss(preds, y, reduction="none").sum(1).mean()
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer

    def get_main_validation_metric(self):
        return {"name": "val_loss", "mode": "min"}

    def get_similar_embedding(self, index):
        item_matrix = self.model.dec.weight.numpy()
        x = item_matrix[index].reshape(1, -1)
        cosine_sim = cosine_similarity(x, item_matrix)
        return np.argsort(-cosine_sim)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--load_checkpoint", type=bool)
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER)
        parser.add_argument("--lr", type=float, default=LR)
        return parser
