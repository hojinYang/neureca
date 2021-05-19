import pickle
import json
from pathlib import Path
from neureca.recommender.training.run_experiment import _import_class
import torch
import numpy as np
import pandas as pd

ITEM_ID_DICT_PATH = (
    Path(__file__).resolve().parents[2] / "demo-toronto" / "preprocessed" / "item_id_dict.json"
)

RATINGS_PATH = Path(__file__).resolve().parents[2] / "demo-toronto" / "data" / "ratings.csv"

CF_PATH = (
    Path(__file__).resolve().parents[0]
    / "training"
    / "logs"
    / "UserBased"
    / "default"
    / "version_0"
    / "checkpoints"
)


def load_model(dir_args_and_checkpoint: Path):

    with open(str(dir_args_and_checkpoint / "args.pkl"), "rb") as f:
        args = pickle.load(f)

    data_class = _import_class(f"neureca.recommender.data.{args.data_class}")
    model_class = _import_class(f"neureca.recommender.{args.model_type}.{args.model_class}")

    data = data_class(args=args)

    ckpt_path = list(dir_args_and_checkpoint.glob("*.ckpt"))[0]
    model = model_class.load_from_checkpoint(ckpt_path, data_config=data.config(), args=args)

    model.eval()
    model.freeze()

    return model


class Recommender:
    def __init__(self):
        with open(str(ITEM_ID_DICT_PATH), "rb") as f:
            self.item2id = json.load(f)
            self.id2item = {v: k for k, v in self.item2id.items()}

        self.item2name = dict(pd.read_csv(str(RATINGS_PATH))[["business_id", "name"]].values)
        self.name2item = {v.lower(): k for k, v in self.item2name.items()}

        self.cf_model = load_model(CF_PATH)

    def convert_name_to_item(self, name_list):
        return [self.name2item[name] for name in name_list]

    def convert_item_to_name(self, item_list):
        return [self.item2name[item] for item in item_list]

    def _get_cf_topK(self, item_list, topK):
        item_id_list = [self.item2id[item] for item in item_list]
        print(item_id_list)

        history = torch.zeros(len(self.item2id))
        history[item_id_list] = 5.0
        history.unsqueeze_(0)
        print(history.sum())

        preds = self.cf_model(history)
        preds = preds.numpy().squeeze()
        print(preds[:50])
        topKs = np.argpartition(preds, -topK)[-topK:]
        topKs = topKs[np.argsort(preds[topKs])]

        return_items = [self.id2item[ind] for ind in topKs]
        return return_items

    def run(self, item_list, topK=25):
        item_recs = self._get_cf_topK(item_list, topK)

        return item_recs

    def get_cosine_sim(self, item):
        id = self.item2id[item]
        lst = self.cf_model.get_similar_embedding(id).squeeze()
        print(lst)
        return_items = [self.id2item[ind] for ind in lst[:10]]
        return return_items


if __name__ == "__main__":
    recommender = Recommender()

    r = pd.read_csv(str(RATINGS_PATH))
    r = r[r["user_id"] == "4xyYBC5MIe-GfDj6kAdAFw"]
    v = list(r["name"].unique())
    v = ["Kekou Gelato House"]

    item_list = recommender.convert_name_to_item(v)
    recs = recommender.run(item_list)
    name_recs = recommender.convert_item_to_name(recs)
    print(name_recs)

    output = recommender.get_cosine_sim(item_list[0])
    name_recs = recommender.convert_item_to_name(output)
    print(name_recs)
