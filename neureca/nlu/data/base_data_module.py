from pathlib import Path
import argparse
import pickle
from typing import Tuple
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
from neureca.nlu.data.util import BaseDataset, NLUYamlToTrainConverter


DATA_DIRNAME = Path(__file__).resolve().parents[3] / "demo-toronto" / "data"
ATTRIBUTE_FILE = DATA_DIRNAME / "attribute.yaml"
NLU_FILE = DATA_DIRNAME / "nlu.yaml"
RATING_FILE = DATA_DIRNAME / "ratings.csv"
TRAIN_DATA_DIRNAME = Path(__file__).resolve().parents[3] / "demo-toronto" / "preprocessed"
RATIO_TRAIN, RATIO_VALID, RATIO_TEST = 0.6, 0.2, 0.2
BATCH_SIZE = 64
NUM_WORKERS = 1


class BaseDataModule(pl.LightningDataModule):
    """
    Base LightningDataModule
    """

    def __init__(self, featurizer, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.ratio_train = self.args.get("ratio_train", RATIO_TRAIN)
        self.ratio_valid = self.args.get("ratio_valid", RATIO_VALID)
        self.ratio_test = self.args.get("ratio_test", RATIO_TEST)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        self.dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.data_train: BaseDataset
        self.data_val: BaseDataset
        self.data_test: BaseDataset

        self.featurizer = featurizer

        self.prepare_data()

    @classmethod
    def data_dirname(cls):
        return DATA_DIRNAME

    @classmethod
    def train_data_dirname(cls):
        return TRAIN_DATA_DIRNAME

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--on_gpu", type=int)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--ratio_train", type=float, default=0.6)
        parser.add_argument("--ratio_valid", type=float, default=0.2)
        parser.add_argument("--ratio_test", type=float, default=0.2)
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {"input_dims": self.dims, "output_dims": self.output_dims}

    def prepare_data(self):
        print("prepare_data")
        if (self.train_data_dirname() / "train.pkl").exists():
            return

        self.train_data_dirname().mkdir(exist_ok=True)
        converter = NLUYamlToTrainConverter(NLU_FILE, ATTRIBUTE_FILE, RATING_FILE)
        converter.update_attribute_dict()
        training_data = converter.convert()
        print(training_data)
        with open(str(self.train_data_dirname() / "train.pkl"), "wb") as f:
            pickle.dump(training_data, f)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
