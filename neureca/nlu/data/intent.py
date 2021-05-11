import argparse
from typing import Optional
import pickle

import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from neureca.nlu.data.base_data_module import BaseDataModule
from neureca.nlu.data.util import BaseDataset


class Intent(BaseDataModule):
    def __init__(self, featurizer, args: argparse.Namespace = None):
        super().__init__(featurizer, args)
        with open(str(self.train_data_dirname() / "train.pkl"), "rb") as f:
            data = pickle.load(f)

        intent_list = data["intents"]

        self.input_dims = self.featurizer.feature_dims
        self.output_dims = len(intent_list)

    def config(self):
        conf = {
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "batch_size": self.batch_size,
        }

        return conf

    def prepare_data(self):
        super().prepare_data()
        print("prepare_data")
        if (self.train_data_dirname() / "intent.pkl").exists():
            return

        with open(str(self.train_data_dirname() / "train.pkl"), "rb") as f:
            data = pickle.load(f)

        intent_list = data["intents"]
        intent_mapper = {v: k for k, v in enumerate(intent_list)}

        X = np.array(
            [self.featurizer.featurize(datum["text"])["features"] for datum in data["examples"]]
        )
        y = np.array([intent_mapper[datum["intent"]] for datum in data["examples"]])

        data = {"X": X, "y": y}

        with open(str(self.train_data_dirname() / "intent.pkl"), "wb") as f:
            pickle.dump(data, f)

    def setup(self, stage: Optional[str] = None) -> None:
        print("data setup")
        with open(str(self.train_data_dirname() / "intent.pkl"), "rb") as f:
            data = pickle.load(f)

        X, y = data["X"], data["y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.ratio_test)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.ratio_valid / (self.ratio_train + self.ratio_valid)
        )

        self.data_train = BaseDataset(X_train, y_train)
        self.data_val = BaseDataset(X_val, y_val)
        self.data_test = BaseDataset(X_test, y_test)
