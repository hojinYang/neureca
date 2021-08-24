import argparse
import torch.nn as nn
from .base_model import BaseModel

LSTM_DIM = 512
LSTM_LAYERS = 1
LSTM_DROPOUT = 0.2


class LSTM(BaseModel):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__(data_config, args)
        lstm_dim = self.args.get("lstm_dim", LSTM_DIM)
        lstm_layers = self.args.get("lstm_layers", LSTM_LAYERS)
        lstm_dropout = self.args.get("lstm_dropout", LSTM_DROPOUT)

        self.lstm = nn.LSTM(
            input_size=self.input_dims[1],
            hidden_size=lstm_dim,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_dim, self.output_dims)

    def forward(self, x):

        # x -> (B, S, E)
        B, S, _E = x.shape
        x = x.permute(1, 0, 2)  # -> (S, B, E)
        x, _ = self.lstm(x)  # -> (S, B, 2 * H) where H is lstm_dim

        # Sum up both directions of the LSTM:
        x = x.view(S, B, 2, -1).sum(dim=2)  # -> (S, B, H)

        x = self.fc(x)  # -> (S, B, C)

        return x.permute(1, 0, 2)  # -> (B, S, C)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--lstm_dim", type=int, default=LSTM_DIM)
        parser.add_argument("--lstm_layers", type=int, default=LSTM_LAYERS)
        parser.add_argument("--lstm_dropout", type=int, default=LSTM_DROPOUT)

        return parser