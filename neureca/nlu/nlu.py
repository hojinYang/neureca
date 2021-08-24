from typing import Optional
import pickle
from pathlib import Path
import torch
import numpy as np
from neureca.shared.utils import import_class, find_latest_subdir
from neureca.nlu.data.utils import convert_uttr_to_attr_dict


def _load_model(args_and_cpkt_path: Path):

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
    def __init__(
        self, intent_version: Optional[str] = None, attribute_version: Optional[str] = None
    ):
        self._intent_version = intent_version
        self._attribute_version = attribute_version

    def load_model(self, path: Path):

        nlu_converted_data_path = path / "preprocessed" / "nlu_converted.pkl"
        if not nlu_converted_data_path.exists():
            raise FileNotFoundError("NLU preprocessed file is not found... Train nlu model first.")

        with open(str(nlu_converted_data_path), "rb") as f:
            nlu_converted_data = pickle.load(f)
        self.intents = nlu_converted_data["intents"]
        self.attribute_tags = nlu_converted_data["attribute_tags"]
        self.item_name = nlu_converted_data["item_name"]

        intent_path = path / "weights" / "Intent"
        if not intent_path.exists():
            raise FileNotFoundError(
                "Intent dir is not found... Train intent classification model first."
            )

        if self._intent_version is None:
            self._intent_version = find_latest_subdir(intent_path)
            print(
                f"The latest version of intent model({self._intent_version}) is loaded as its version is not specified."
            )
        intent_path = intent_path / self._intent_version
        if not intent_path.exists():
            raise FileNotFoundError(f"{intent_path} is not found.")

        attribute_path = path / "weights" / "Attribute"
        if not attribute_path.exists():
            raise FileNotFoundError(
                "Attribute dir is not found... Train attribute recognizer model first."
            )
        if self._attribute_version is None:
            self._attribute_version = find_latest_subdir(attribute_path)
            print(
                f"The latest version of attribute model({self._attribute_version}) is loaded as its version is not specified."
            )
        attribute_path = attribute_path / self._attribute_version
        if not intent_path.exists():
            raise FileNotFoundError(f"{attribute_path} is not found.")

        self.intent_model, self.intent_feat, self.intent_args = _load_model(intent_path)
        self.attribute_model, self.attribute_feat, self.attribute_args = _load_model(attribute_path)

    def run(self, uttr):
        intent = self._get_intent(uttr)
        attrs, items = self._get_attributes(uttr)
        output = NLUOutput(uttr, intent, attrs, items)
        # print(output)
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
        attrs = convert_uttr_to_attr_dict(uttr, bio_tags, offsets, self.attribute_tags)
        items = attrs.pop(self.item_name, None)

        return attrs, items


class NLUOutput:
    def __init__(self, uttr, intent, attributes, items):
        self.intent = intent
        self.uttr = uttr
        self.attributes = attributes
        self.items = items

    def __str__(self):
        return f"intent-> {self.intent}\n attributes-> {self.attributes}\n itmes-> {self.items} \n uttr-> {self.uttr}"
