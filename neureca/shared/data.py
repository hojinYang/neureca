import argparse
from pathlib import Path
from typing import Tuple, Union, Sequence, Callable, Any
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader


DATA_DIRNAME = "data"
PREPROCESSED_DIRNAME = "preprocessed"

ATTRIBUTE_FILENAME = "attribute.yaml"
NLU_FILENAME = "nlu.yaml"
RATING_FILENAME = "ratings.csv"


RATIO_TRAIN, RATIO_VALID, RATIO_TEST = 0.6, 0.2, 0.2
BATCH_SIZE = 64
NUM_WORKERS = 1

SequenceOrTensor = Union[Sequence, torch.tensor]


class BaseDataModule(pl.LightningDataModule):
    """
    Base LightningDataModule
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.ratio_train = self.args.get("ratio_train", RATIO_TRAIN)
        self.ratio_valid = self.args.get("ratio_valid", RATIO_VALID)
        self.ratio_test = self.args.get("ratio_test", RATIO_TEST)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)

        path = Path(self.args.get("path"))
        data_dirname = path / DATA_DIRNAME
        self.attribute_path = data_dirname / ATTRIBUTE_FILENAME
        self.nlu_path = data_dirname / NLU_FILENAME
        self.rating_path = data_dirname / RATING_FILENAME
        self.prepocessed_dirname = path / PREPROCESSED_DIRNAME

        self.input_dim: Tuple[int, ...]
        self.output_dim: Tuple[int, ...]

        self.data_train: BaseDataset
        self.data_val: BaseDataset
        self.data_test: BaseDataset

    def prepare_data(self):
        raise NotImplementedError

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

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--on_gpu", type=int)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--ratio_train", type=float, default=RATIO_TRAIN)
        parser.add_argument("--ratio_valid", type=float, default=RATIO_VALID)
        parser.add_argument("--ratio_test", type=float, default=RATIO_TEST)
        parser.add_argument("--path", type=str, default=str(Path.cwd()))
        return parser


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.
    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(targets) != len(data):
            raise ValueError("Data and targets must be of equal length")

        self.data = data
        self.targets = targets
        self.data_transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return length of the dataset"""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by trasform function
        """
        datum, target = self.data[index], self.targets[index]

        if self.data_transform is not None:
            datum = self.data_transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target