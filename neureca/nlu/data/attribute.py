import argparse
from typing import Optional
import pickle

import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from neureca.nlu.data.base_data_module import BaseDataModule
from neureca.nlu.data.util import BaseDatasetWithMask


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
            output = self.featurizer.featurize(datum["text"])

            bio_tags = self.featurizer.get_bio_tags(
                datum["attributes"], attribute_mapper, output["offset_mapping"]
            )
            X.append(output["features"])
            y.append(bio_tags)
            mask.append(output["mask"])

        data = {"X": np.array(X), "y": np.array(y), "mask": np.array(mask)}

        with open(str(self.train_data_dirname() / "attribute.pkl"), "wb") as f:
            pickle.dump(data, f)

    def setup(self, stage: Optional[str] = None) -> None:
        print("data setup")
        with open(str(self.train_data_dirname() / "attribute.pkl"), "rb") as f:
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

        self.data_train = BaseDatasetWithMask(X_train, y_train, mask_train)
        self.data_val = BaseDatasetWithMask(X_val, y_val, mask_val)
        self.data_test = BaseDatasetWithMask(X_test, y_test, mask_test)
