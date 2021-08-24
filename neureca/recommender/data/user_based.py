import argparse
from pathlib import Path
import random
import json
from typing import Sequence, Union, Callable, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from neureca.shared.data import BaseDataModule, BaseDataset
from neureca.recommender.data.utils import csr_to_array, df_to_sparse, preprocess_rating

SequenceOrTensor = Union[Sequence, torch.tensor]

RATING_DATA = "ratings.csv"
ITEM_ID_DICT = "item_id_dict.json"
USER_ID_DICT = "user_id_dict.json"


class UserBased(BaseDataModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        rating_data = Path(self.args.get("rating_data", RATING_DATA))
        item_id_dict = Path(self.args.get("item_id_dict", ITEM_ID_DICT))
        user_id_dict = Path(self.args.get("user_id_dict", USER_ID_DICT))

        self.rating_data_path = self.prepocessed_dirname / rating_data
        self.user_id_dict_path = self.prepocessed_dirname / user_id_dict
        self.item_id_dict_path = self.prepocessed_dirname / item_id_dict

        self.prepare_data()

        with open(str(self.item_id_dict_path), "rb") as f:
            data = json.load(f)
            self.num_items = len(data)
        with open(str(self.user_id_dict_path), "rb") as f:
            data = json.load(f)
            self.num_users = len(data)

    def config(self):
        conf = {"input_dims": self.num_items, "output_dims": self.num_items}

        return conf

    def prepare_data(self):
        if not self.rating_data_path.exists():
            self.rating_data_path.resolve().parents[0].mkdir(exist_ok=True)
            preprocess_rating(
                self.rating_path,
                self.rating_data_path,
                self.user_id_dict_path,
                self.item_id_dict_path,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        print("data setup")
        with open(str(self.rating_data_path), "rb") as f:
            ratings = pd.read_csv(f)

        M_train, M_val, M_test = self._split_data(ratings)

        self.data_train = SparseDataset(M_train, M_train, csr_to_array, csr_to_array)
        self.data_val = SparseDataset(M_train, M_val, csr_to_array, csr_to_array)
        self.data_test = SparseDataset(M_train, M_test, csr_to_array, csr_to_array)

    def _split_data(self, ratings):

        """
        Proprocess user-item-rating raw data into csr matrix.
        Split tr/val/te
        """

        user_group = ratings.groupby("user")

        tr_data, val_data, te_data = list(), list(), list()

        for _, group in tqdm(user_group):
            num_items_user = len(group)

            assigned_set = np.zeros(num_items_user, dtype=int)

            idx = list(range(num_items_user))
            random.shuffle(idx)
            train_offset = int(num_items_user * self.ratio_train)
            valid_offset = int(num_items_user * (self.ratio_train + self.ratio_valid))

            assigned_set[idx[:train_offset]] = 0
            assigned_set[idx[train_offset:valid_offset]] = 1
            assigned_set[idx[valid_offset:]] = 2

            group["set"] = assigned_set

            tr_data.append(group[group["set"] == 0])
            val_data.append(group[group["set"] == 1])
            te_data.append(group[group["set"] == 2])

        shape = (self.num_users, self.num_items)
        tr_data = df_to_sparse(pd.concat(tr_data), shape)
        val_data = df_to_sparse(pd.concat(val_data), shape)
        te_data = df_to_sparse(pd.concat(te_data), shape)

        return tr_data, val_data, te_data

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--rating_data", type=str, default=RATING_DATA)
        parser.add_argument("--user_id_dict", type=str, default=USER_ID_DICT)
        parser.add_argument("--item_id_dict", type=str, default=ITEM_ID_DICT)
        return parser


class SparseDataset(BaseDataset):
    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:

        if data.shape != targets.shape:
            raise ValueError("Data and targets must be of equal shape")

        self.data = data
        self.targets = targets
        self.data_transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]