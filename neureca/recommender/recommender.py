import pickle
import json
from pathlib import Path
from neureca.recommender.training.run_experiment import _import_class
import torch
import numpy as np

ITEM_ID_DICT_PATH = (
    Path(__file__).resolve().parents[2] / "demo-toronto" / "preprocessed" / "item_id_dict.json"
)

CF_PATH = (
    Path(__file__).resolve().parents[0]
    / "training"
    / "logs"
    / "Intent"
    / "default"
    / "version_2"
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

        self.cf_model = load_model(CF_PATH)

    def _get_cf_topK(self, item_list, topK):
        item_id_list = [self.item2id[item] for item in item_list]
        preds = self.cf_model.pred_from_item_list(item_id_list)
        topKs = np.argpartition(preds, -topK)[-topK:]
        topKs = topK[np.argsort(preds[topKs])]

        return_items = [self.id2item[ind] for ind in topKs]
        return return_items

    def run(self, uttr):
        intent = self._get_intent(uttr)
        attrs = self._get_attributes(uttr)
        output = {"intent": intent, "attributes": attrs}
        return output


if __name__ == "__main__":
    nlu = NLU()
    print(nlu.run("Is there any chinese menu?"))
    print(nlu.run("Is there any japanese menu?"))
    print(nlu.run("I am looking for korean restaurant with descent patio."))
    print(nlu.run("I am looking for indian restaurant with descent patio."))

    # print(TRAIN_DATA_PATH)
    # with
    # path = Path("/home/hojin/code/neureca/neureca/nlu/training/logs/default/version_5/checkpoints")
    # model = load_model(path)
