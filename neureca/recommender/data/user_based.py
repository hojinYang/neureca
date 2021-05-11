import argparse
from typing import Optional
import pickle
import random
import json

import pytorch_lightning as pl
import numpy as np
import pandas as pd
from tqdm import tqdm

from neureca.recommender.data.base_data_module import BaseDataModule
from neureca.recommender.data.util import SparseDataset
from neureca.recommender.data.util import csr_to_array, df_to_sparse


class UserBased(BaseDataModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args=args)

        with open(str(self.train_data_dirname() / "item_id_dict.json"), "rb") as f:
            data = json.load(f)
            self.num_items = len(data)
        with open(str(self.train_data_dirname() / "user_id_dict.json"), "rb") as f:
            data = json.load(f)
            self.num_users = len(data)
        print(self.num_users, self.num_items)

    def config(self):
        conf = {
            "input_dims": self.num_items,
        }

        return conf

    def prepare_data(self):
        super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        print("data setup")
        with open(str(self.train_data_dirname() / "ratings.csv"), "rb") as f:
            ratings = pd.read_csv(f)

        M_train, M_val, M_test = self._split_data(ratings)

        self.data_train = SparseDataset(M_train, M_train, csr_to_array, csr_to_array)
        self.data_val = SparseDataset(M_train, M_val, csr_to_array, csr_to_array)
        self.data_test = SparseDataset(M_train, M_test, csr_to_array, csr_to_array)

    def _split_data(self, ratings):

        """
        Proprocess UIRT raw data into trainable form.
        Holdout feedbacks for test per user.
        Save preprocessed data.
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
