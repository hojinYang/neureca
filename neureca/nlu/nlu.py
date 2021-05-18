import pickle
from pathlib import Path
from neureca.nlu.training.run_experiment import _import_class
import torch
import numpy as np

TRAIN_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "demo-toronto" / "preprocessed" / "train.pkl"
)

INTENT_PATH = (
    Path(__file__).resolve().parents[0]
    / "training"
    / "logs"
    / "Intent"
    / "default"
    / "version_0"
    / "checkpoints"
)

ATTRIBUTE_PATH = (
    Path(__file__).resolve().parents[0]
    / "training"
    / "logs"
    / "Attribute"
    / "default"
    / "version_1"
    / "checkpoints"
)


def load_model(dir_args_and_checkpoint: Path):

    with open(str(dir_args_and_checkpoint / "args.pkl"), "rb") as f:
        args = pickle.load(f)

    data_class = _import_class(f"neureca.nlu.data.{args.data_class}")
    model_class = _import_class(f"neureca.nlu.{args.model_type}.{args.model_class}")
    feat_class = _import_class(f"neureca.nlu.featurizers.{args.featurizer_class}")
    featurizer = feat_class(args)
    data = data_class(featurizer=featurizer, args=args)

    ckpt_path = list(dir_args_and_checkpoint.glob("*.ckpt"))[0]
    model = model_class.load_from_checkpoint(ckpt_path, data_config=data.config(), args=args)

    model.eval()
    model.freeze()

    return model, featurizer


class NLU:
    def __init__(self):
        with open(str(TRAIN_DATA_PATH), "rb") as f:
            data = pickle.load(f)
        self.intents = data["intents"]
        self.attributes = data["attributes"]

        self.intent_model, self.intent_feat = load_model(INTENT_PATH)
        self.attribute_model, self.attribute_feat = load_model(ATTRIBUTE_PATH)

    def run(self, uttr):
        intent = self._get_intent(uttr)
        attrs = self._get_attributes(uttr)
        output = {"intent": intent, "attributes": attrs}
        return output

    def _get_intent(self, uttr):
        output = self.intent_feat.featurize(uttr)

        feats = output["features"]
        intent_idx = self.intent_model(torch.tensor([feats])).squeeze().numpy()

        return self.intents[np.argmax(intent_idx)]

    def _get_attributes(self, uttr):
        output = self.attribute_feat.featurize(uttr)

        feats = output["features"]
        mask = output["mask"]
        offsets = output["offset_mapping"]

        bio_tags = self.attribute_model.decode(torch.tensor([feats]), torch.tensor([mask]))[0]
        ret = self.attribute_feat.get_attributes(uttr, bio_tags, offsets, self.attributes)

        return ret


if __name__ == "__main__":
    nlu = NLU()
    print(nlu.run("Is there any chinese menu?"))
    print(nlu.run("Is there any japanese menu?"))
    print(nlu.run("I am looking for korean restaurant with descent patio."))
    print(nlu.run("I'm looking for a place for dinner in annex with my girlfriend"))
    print(
        nlu.run(
            "I'm looking for a place for like BukChangDongSoonTofu for lunch in annex with my girlfriend"
        )
    )
    print(
        nlu.run(
            "I'm looking for a place for like Buk ChangDongSoonTofu or Sushi on Bloor for family gathering"
        )
    )
    print(nlu.run("Can you recommend a restaurant for me"))
    print(nlu.run("You know, sushi is always right"))
    print(nlu.run("You know, korean food is always right"))
    print(nlu.run("You know, coffee is always right"))
    print(nlu.run("I like the place like Tim Hortons or Starbucks"))
