import argparse
from typing import Optional
import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

from neureca.nlu.data.base_nlu_data import BaseNLUDataModule
from neureca.nlu.featurizers import BaseFeaturizer
from neureca.shared.data import BaseDataset

INTENT_DATA = "intent.pkl"
USE_SENTENCE_EMB = True


class Intent(BaseNLUDataModule):
    def __init__(self, featurizer: BaseFeaturizer, args: argparse.Namespace = None):
        super().__init__(args)

        intent_data = Path(self.args.get("intent_data", INTENT_DATA))
        self.intent_data_path = self.prepocessed_dirname / intent_data
        self.featurizer = featurizer
        self.use_sentence_emb = self.args.get("use_sentence_emb", USE_SENTENCE_EMB)

        self.input_dims = self.featurizer.feature_dims[0] * self.featurizer.feature_dims[1]
        if self.use_sentence_emb:
            self.input_dims = self.featurizer.feature_dims[1]

        self.output_dims = self.num_intents

    def config(self):
        conf = {
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
        }

        return conf

    def prepare_data(self):
        if self.intent_data_path.exists():
            return

        with open(str(self.nlu_converted_data_path), "rb") as f:
            data = pickle.load(f)

        intent_list = data["intents"]
        intent_mapper = {v: k for k, v in enumerate(intent_list)}

        X = np.array(
            [
                self.featurizer.featurize(datum["text"], self.use_sentence_emb)["features"]
                for datum in data["examples"]
            ]
        )
        y = np.array([intent_mapper[datum["intent"]] for datum in data["examples"]])

        intent_data = {"X": X, "y": y}

        with open(str(self.intent_data_path), "wb") as f:
            pickle.dump(intent_data, f)

    def setup(self, stage: Optional[str] = None) -> None:

        with open(str(self.intent_data_path), "rb") as f:
            data = pickle.load(f)

        X, y = data["X"], data["y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.ratio_test)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=self.ratio_valid / (self.ratio_train + self.ratio_valid)
        )

        self.data_train = BaseDataset(X_train, y_train)
        self.data_val = BaseDataset(X_val, y_val)
        self.data_test = BaseDataset(X_test, y_test)

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseNLUDataModule.add_to_argparse(parser)
        parser.add_argument("--intent_data", type=str, default=INTENT_DATA)
        parser.add_argument("--use_sentence_emb", type=bool, default=USE_SENTENCE_EMB)
        return parser
