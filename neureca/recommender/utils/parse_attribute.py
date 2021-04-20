"""
Parse attributes
"""
from pathlib import Path
from typing import Dict, List, Union, Sequence
import re
import yaml
import pandas as pd
from tqdm import tqdm
from transformers import pipeline


class Attribute:
    """
    class Attribute
    """

    def __init__(self, d: Dict[str, str]):
        self.name = d["attr"]
        self.syn = d["syn"]
        self.regex_exp = r"\b(" + "|".join(w for w in self.syn) + ")" + r"\b"

    def check(self, sentence: str) -> bool:
        """
        check if
        """
        return re.search(self.regex_exp, sentence) is not None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class AttributeParser:
    """
    Attribute parser
    """

    def __init__(self):
        self.attributes: List["Attribute"] = list()
        self.classifier = pipeline("sentiment-analysis")

    def generate_attribute(self, attribute_path: Path) -> None:
        with open(attribute_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        st = data["attribute"]

        while len(st) > 0:
            attr = st.pop()
            if "sub-attr" in attr.keys():
                for s in attr["sub-attr"]:
                    st.append(s)
            else:
                self.attributes.append(Attribute(attr))

    def _read_single_review_line(self, line: str) -> List[str]:
        relevant_attributes = list()
        for a in self.attributes:
            if a.check(line):
                relevant_attributes.append(a.name)
        return relevant_attributes

    def _read_single_review(
        self, review_text_path: Path
    ) -> Dict[str, List[Union[int, float, str]]]:
        review_dict: Dict[str, List[Union[int, float, str]]] = {
            "uid": list(),
            "iid": list(),
            "review_line": list(),
            "attr": list(),
            "sentiment": list(),
        }

        uid, iid = review_text_path.stem.split("-")

        with open(review_text_path, "r") as f:
            for line in f.readlines():
                line = re.sub(";", "", line.lower().strip())
                review_dict["attr"] = list(self._read_single_review_line(line))
                review_dict["review_line"] = [line] * len(review_dict["attr"])

        review_dict["uid"] = [int(uid)] * len(review_dict["attr"])
        review_dict["iid"] = [int(iid)] * len(review_dict["attr"])

        if len(review_dict["review_line"]) > 0:
            sents = self.classifier(review_dict["review_line"])
            review_dict["sentiment"] = [
                x["score"] if x["label"] == "POSITIVE" else -x["score"]
                for x in sents
            ]

        return review_dict

    def build_dataset(self, review_path: Path) -> None:
        review_dict: Dict[str, List[Union[int, float, str]]] = {
            "uid": list(),
            "iid": list(),
            "review_line": list(),
            "attr": list(),
            "sentiment": list(),
        }

        for review_text_path in tqdm(review_path.glob("*.txt")):
            _review_dict = self._read_single_review(review_text_path)
            for key in review_dict:
                review_dict[key] += _review_dict[key]

        ret = pd.DataFrame(review_dict)
        ret.to_csv("temp.csv", index=False)
