from typing import Optional
import pickle
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from neureca.shared.utils import import_class, find_latest_subdir


def _load_model(args_and_cpkt_path: Path):

    with open(str(args_and_cpkt_path / "args.pkl"), "rb") as f:
        args = pickle.load(f)

    data_class = import_class(f"neureca.recommender.data.{args.data_class}")
    model_class = import_class(f"neureca.shared.models.{args.model_class}")
    lit_wrapper_class = import_class(f"neureca.recommender.lit_wrappers.{args.lit_wrapper_class}")

    data = data_class(args=args)
    model = model_class(data_config=data.config(), args=args)

    ckpt_path = list((args_and_cpkt_path / "checkpoints").glob("*.ckpt"))[0]

    lit_wrapper = lit_wrapper_class.load_from_checkpoint(ckpt_path, model=model, args=args)
    lit_wrapper.eval()
    lit_wrapper.freeze()

    return lit_wrapper, args


class Recommender:
    def __init__(self, version: Optional[str] = None):
        self._version = version

    def load_model(self, path: Path):
        rec_path = path / "weights" / "UserBased"
        if not rec_path.exists():
            raise FileNotFoundError(
                "recommender dir is not found... Train recommender model first."
            )

        if self._version is None:

            self._version = find_latest_subdir(rec_path)
            print(
                f"The latest version of recommender model({self._version}) is loaded as its version is not specified."
            )
        rec_path = rec_path / self._version
        if not rec_path.exists():
            raise FileNotFoundError(f"{rec_path} is not found.")

        self.rec_model, self.rec_args = _load_model(rec_path)

        item_id_dict_path = path / "preprocessed" / self.rec_args.item_id_dict
        with open(str(item_id_dict_path), "rb") as f:
            self.item2id = json.load(f)
            self.id2item = {v: k for k, v in self.item2id.items()}

        original_rating_path = path / "data" / "ratings.csv"
        self.item2name = dict(
            pd.read_csv(str(original_rating_path))[["business_id", "name"]].values
        )
        self.name2item = {v.lower(): k for k, v in self.item2name.items()}

    def convert_name_to_item(self, names):
        if isinstance(names, list):
            return [self.name2item[name.lower()] for name in names]
        elif isinstance(names, str):
            return self.name2item[names.lower()]
        else:
            raise NotImplementedError()

    def convert_item_to_name(self, items):
        if isinstance(items, list):
            return [self.item2name[item] for item in items]
        elif isinstance(items, str):
            return self.item2name[items]
        else:
            raise NotImplementedError()

    def _get_cf_topK(self, item_list, topK):
        item_id_list = [self.item2id[item] for item in item_list]

        history = torch.zeros(len(self.item2id))
        history[item_id_list] = 5.0
        history.unsqueeze_(0)

        preds = self.rec_model(history)
        preds = preds.numpy().squeeze()

        topKs = np.argpartition(preds, -topK)[-topK:]
        topKs = topKs[np.argsort(preds[topKs])]

        return_items = [self.id2item[ind] for ind in topKs]
        return return_items

    def run(self, item_list, topK=25):
        if not isinstance(item_list, list):
            item_list = [item_list]
        item_recs = self._get_cf_topK(item_list, topK)

        return item_recs
