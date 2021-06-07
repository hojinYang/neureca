import pickle
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd

from neureca.shared.utils import import_class


def load_model(args_and_cpkt_path: Path):

    with open(str(args_and_cpkt_path / "args.pkl"), "rb") as f:
        args = pickle.load(f)
    print(args)

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
    def __init__(self, demo_path: Path, recommender_version: str):

        rec_path = demo_path / "weights" / "UserBased" / recommender_version
        self.rec_model, self.rec_args = load_model(rec_path)
        print(self.rec_args)
        item_id_dict_path = (
            demo_path / self.rec_args.preprocessed_dirname / self.rec_args.item_id_dict
        )
        with open(str(item_id_dict_path), "rb") as f:
            self.item2id = json.load(f)
            self.id2item = {v: k for k, v in self.item2id.items()}

        original_rating_path = (
            demo_path / self.rec_args.data_dirname / self.rec_args.rating_filename
        )
        self.item2name = dict(
            pd.read_csv(str(original_rating_path))[["business_id", "name"]].values
        )
        self.name2item = {v.lower(): k for k, v in self.item2name.items()}

    def convert_name_to_item(self, name_list):

        return [self.name2item[name.lower()] for name in name_list]

    def convert_item_to_name(self, item_list):
        return [self.item2name[item] for item in item_list]

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
        item_recs = self._get_cf_topK(item_list, topK)

        return item_recs

    def get_cosine_sim(self, item):
        id = self.item2id[item]
        lst = self.rec_model.get_similar_embedding(id).squeeze()

        return_items = [self.id2item[ind] for ind in lst[:10]]
        return return_items


DEMO_PATH = Path(__file__).resolve().parents[2] / "demo-toronto"
REC_VERSION = "version_1"
if __name__ == "__main__":
    recommender = Recommender(DEMO_PATH, REC_VERSION)
    original_rating_path = (
        DEMO_PATH / recommender.rec_args.data_dirname / recommender.rec_args.rating_filename
    )
    r = pd.read_csv(str(original_rating_path))
    # r = r[r["user_id"] == "4xyYBC5MIe-GfDj6kAdAFw"]
    # v = list(r["name"].unique())
    v = ["Kekou Gelato House"]
    print(v)

    item_list = recommender.convert_name_to_item(v)
    recs = recommender.run(item_list)
    name_recs = recommender.convert_item_to_name(recs)
    print(name_recs)

    output = recommender.get_cosine_sim(item_list[0])
    name_recs = recommender.convert_item_to_name(output)
    print(name_recs)
