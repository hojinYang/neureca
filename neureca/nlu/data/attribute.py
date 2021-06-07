import argparse
from typing import Optional
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from neureca.shared.data import BaseDataset
from neureca.nlu.data.base_nlu_data import BaseNLUDataModule
from neureca.nlu.data.utils import get_bio_tags


ATTRIBUTE_DATA = "attribute.pkl"


class Attribute(BaseNLUDataModule):
    def __init__(self, featurizer, args: argparse.Namespace = None):
        super().__init__(args)

        attribute_data = Path(self.args.get("attribute_data", ATTRIBUTE_DATA))
        self.attribute_data_path = self.prepocessed_dirname / attribute_data
        self.featurizer = featurizer

        self.input_dim = self.featurizer.feature_dims
        self.output_dim = self.num_attribute_tags

    def config(self):
        conf = {
            "input_dims": self.input_dim,
            "output_dims": self.output_dim,
        }

        return conf

    def prepare_data(self):
        if self.attribute_data_path.exists():
            return

        with open(str(self.nlu_converted_data_path), "rb") as f:
            data = pickle.load(f)

        attribute_tag_mapper = {v: k for k, v in enumerate(data["attribute_tags"])}

        X, y, mask = list(), list(), list()

        for datum in data["examples"]:
            output = self.featurizer.featurize(datum["text"], use_sentence_emb=False)

            bio_tags = get_bio_tags(
                datum["attributes"], output["offset_mapping"], attribute_tag_mapper
            )
            X.append(output["features"])
            y.append(bio_tags)
            mask.append(output["mask"])

        data = {"X": np.array(X), "y": np.array(y), "mask": np.array(mask)}

        with open(str(self.attribute_data_path), "wb") as f:
            pickle.dump(data, f)

    def setup(self, stage: Optional[str] = None) -> None:
        print("data setup")
        with open(str(self.attribute_data_path), "rb") as f:
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


class BaseDatasetWithMask(BaseDataset):
    def __init__(
        self,
        data,
        targets,
        masks,
        transform=None,
        target_transform=None,
    ) -> None:
        super().__init__(data, targets, transform, target_transform)

        if (len(targets) != len(masks)) or (len(data) != len(masks)):
            raise ValueError("Data and targets must be of equal length")

        self.masks = masks

    def __getitem__(self, index: int):
        """
        Return a datum and its target, after processing by trasform function
        """
        datum, target, mask = self.data[index], self.targets[index], self.masks[index]

        if self.data_transform is not None:
            datum = self.data_transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target, mask
