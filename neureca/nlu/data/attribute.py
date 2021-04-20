import argparse
from typing import Optional
import pickle

import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from neureca.nlu.data.base_data_module import BaseDataModule
from neureca.nlu.data.util import BaseDataset


class Attribute(BaseDataModule):
    def __init__(self, featurizer, args: argparse.Namespace = None):
        super().__init__(featurizer, args)
        print("attribute class init")
        self.input_dims = self.featurizer.feature_dims

        with open(str(self.train_data_dirname() / "train.pkl"), "rb") as f:
            data = pickle.load(f)

        attribute_list = data["attributes"]
        self.output_dims = len(attribute_list) * 2 + 1

    def config(self):
        conf = {
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "batch_size": self.batch_size,
        }

        return conf

    def prepare_data(self):
        super().prepare_data()

        if (self.train_data_dirname() / "attribute.pkl").exists():
            return

        with open(str(self.train_data_dirname() / "train.pkl"), "rb") as f:
            data = pickle.load(f)

        attribute_list = data["attributes"]
        attribute_mapper = {v: k for k, v in enumerate(attribute_list)}

        X, y, mask = list(), list(), list()
        for datum in data["examples"]:
            X_, y_, mask_ = self.featurizer.get_bio_tags(
                datum["text"], datum["attributes"], attribute_mapper
            )
            X.append(X_)
            y.append(y_)
            mask.append(mask_)
        data = {"X": np.array(X), "y": np.array(y), "mask": np.array(mask)}

        with open(str(self.train_data_dirname() / "attribute.pkl"), "wb") as f:
            pickle.dump(data, f)

    def setup(self, stage: Optional[str] = None) -> None:
        print("data setup")
        with open(str(self.train_data_dirname() / "intent.pkl"), "rb") as f:
            data = pickle.load(f)

        X, y, mask = data["X"], data["y"], data["mask"]
        X_train, X_test, y_train, y_test, mask_train, mask_test = train_test_split(
            X, y, mask, test_size=self.ratio_test
        )
        X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(
            X_train,
            y_train,
            mask_train,
            test_size=self.ratio_valid / (self.ratio_train + self.ratio_valid),
        )

        self.data_train = BaseDataset(X_train, y_train)
        self.data_val = BaseDataset(X_val, y_val)
        self.data_test = BaseDataset(X_test, y_test)


if __name__ == "__main__":
    from neureca.nlu.featurizers.bert import Bert

    feat = Bert()
    a = Attribute(feat)