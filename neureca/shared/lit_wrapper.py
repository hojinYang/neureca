import argparse
from pytorch_lightning.core.lightning import LightningModule
from neureca.shared.models import base_model


class BaseLitWrapper(LightningModule):
    def __init__(self, model: base_model, args: argparse.Namespace = None):
        super().__init__()

        self.args = vars(args) if args is not None else {}
        self.model = model

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_index):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def get_main_validation_metric(self):
        raise NotImplementedError
