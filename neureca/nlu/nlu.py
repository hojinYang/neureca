import pickle
from pathlib import Path
import torch
import numpy as np
from neureca.shared.utils import import_class
from neureca.nlu.data.utils import convert_uttr_to_attr_dict


def load_model(args_and_cpkt_path: Path):

    with open(str(args_and_cpkt_path / "args.pkl"), "rb") as f:
        args = pickle.load(f)

    data_class = import_class(f"neureca.nlu.data.{args.data_class}")
    model_class = import_class(f"neureca.shared.models.{args.model_class}")
    lit_wrapper_class = import_class(f"neureca.nlu.lit_wrappers.{args.lit_wrapper_class}")
    feat_class = import_class(f"neureca.nlu.featurizers.{args.featurizer_class}")

    featurizer = feat_class(args)
    data = data_class(featurizer=featurizer, args=args)
    model = model_class(data_config=data.config(), args=args)

    ckpt_path = list((args_and_cpkt_path / "checkpoints").glob("*.ckpt"))[0]

    lit_wrapper = lit_wrapper_class.load_from_checkpoint(ckpt_path, model=model, args=args)
    lit_wrapper.eval()
    lit_wrapper.freeze()

    return lit_wrapper, featurizer, args


class NLU:
    def __init__(self, demo_path: Path, intent_version: str, attribute_version: str):
        nlu_converted_data_path = demo_path / "preprocessed" / "nlu_converted.pkl"
        with open(str(nlu_converted_data_path), "rb") as f:
            nlu_converted_data = pickle.load(f)
        self.intents = nlu_converted_data["intents"]
        self.attribute_tags = nlu_converted_data["attribute_tags"]

        intent_path = demo_path / "weights" / "Intent" / intent_version
        attribute_path = demo_path / "weights" / "Attribute" / attribute_version

        self.intent_model, self.intent_feat, self.intent_args = load_model(intent_path)
        self.attribute_model, self.attribute_feat, self.attribute_args = load_model(attribute_path)

    def run(self, uttr):
        intent = self._get_intent(uttr)
        attrs = self._get_attributes(uttr)
        output = {"uttr": uttr, "intent": intent, "attributes": attrs}

        return output

    def _get_intent(self, uttr):
        output = self.intent_feat.featurize(
            uttr, use_sentence_emb=self.intent_args.use_sentence_emb
        )

        feats = output["features"]
        intent_idx = self.intent_model(torch.tensor([feats])).squeeze().numpy()

        return self.intents[np.argmax(intent_idx)]

    def _get_attributes(self, uttr):
        output = self.attribute_feat.featurize(uttr, use_sentence_emb=False)

        feats = output["features"]
        mask = output["mask"]
        offsets = output["offset_mapping"]

        bio_tags = self.attribute_model.decode(torch.tensor([feats]), torch.tensor([mask]))[0]
        ret = convert_uttr_to_attr_dict(uttr, bio_tags, offsets, self.attribute_tags)

        return ret


DEMO_PATH = Path(__file__).resolve().parents[2] / "demo-toronto"
INTENT_VERSION = "version_1"
ATTRIBUTE_VERSION = "version_2"


if __name__ == "__main__":
    nlu = NLU(DEMO_PATH, INTENT_VERSION, ATTRIBUTE_VERSION)
    print(nlu.run("Is there any chinese menu?"))
    print(nlu.run("Is there any japanese menu?"))
    print(nlu.run("I am looking for korean restaurant with descent patio."))
    print(nlu.run("I'm looking for a place for dinner in annex with my girlfriend"))
    print(
        nlu.run(
            "I'm looking for a place for like Buk Chang Dong Soon Tofu for lunch in annex with my girlfriend"
        )
    )
    print(
        nlu.run(
            "I'm looking for a place for like Buk Chang Dong Soon Tofu or Sushi on Bloor for family gathering"
        )
    )
    print(nlu.run("Can you recommend a restaurant for me"))
    print(nlu.run("You know, sushi is always right"))
    print(nlu.run("You know, korean food is always right"))
    print(nlu.run("You know, coffee is always right"))
    print(nlu.run("I like the place like Tim Hortons or Starbucks"))
