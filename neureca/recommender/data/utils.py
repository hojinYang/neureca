from typing import Sequence, Union, Any, Callable, Tuple, Dict
import pickle
import json

import pandas as pd
import scipy.sparse as sp
import torch

SequenceOrTensor = Union[Sequence, torch.tensor]


def csr_to_array(csr_matrix):
    return csr_matrix.toarray().squeeze()


def preprocess_rating(
    original_rating_path, preprocessed_rating_path, user_id_dict_path, item_id_dict_path
):
    """
    Read raw data.
    """

    print('Loading the dataset from "%s"' % original_rating_path)

    data = pd.read_csv(
        original_rating_path,
        header=0,
        usecols=[0, 1, 2],
        names=["user", "item", "rating"],
        engine="python",
    )

    data, user_id_dict, item_id_dict = _assign_id(data)
    num_users, num_items, num_ratings = len(user_id_dict), len(item_id_dict), len(data)

    data.to_csv(preprocessed_rating_path, index=False)

    info_lines = []
    info_lines.append(
        "# users: %d, # items: %d, # ratings: %d" % (num_users, num_items, num_ratings)
    )
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))

    with open(user_id_dict_path, "w") as f:
        json.dump(user_id_dict, f)

    with open(item_id_dict_path, "w") as f:
        json.dump(item_id_dict, f)

    print("Preprocess finished.")


def _assign_id(data):
    """
    Assign old user/item id into new consecutive ids.
    """

    # initial # user, items
    num_users = len(pd.unique(data.user))
    num_items = len(pd.unique(data.item))

    print("initial user, item:", num_users, num_items)

    user_df = data.groupby("user", as_index=False).size().set_index("user")
    user_df.columns = ["item_cnt"]
    user_df = user_df.sort_values(by="item_cnt", ascending=False)
    user_df["new_id"] = list(range(num_users))

    user_id_dict = user_df.to_dict()["new_id"]
    data.user = [user_id_dict[x] for x in data.user.tolist()]

    item_df = data.groupby("item", as_index=False).size().set_index("item")
    item_df.columns = ["user_cnt"]
    item_df = item_df.sort_values(by="user_cnt", ascending=False)
    item_df["new_id"] = list(range(num_items))

    item_id_dict = item_df.to_dict()["new_id"]
    data.item = [item_id_dict[x] for x in data.item.tolist()]

    return data, user_id_dict, item_id_dict


def save_data_to_sparse(data, num_users, num_items, save_dir):
    sparse = df_to_sparse(data, shape=(num_users, num_items))

    with open(save_dir / "rating.pkl", "wb") as f:
        pickle.dump(sparse, f)


def df_to_sparse(df, shape=None):
    rows, cols = df.user, df.item
    values = df.rating

    sp_data = sp.csr_matrix((values, (rows, cols)), dtype="float32", shape=shape)
    return sp_data