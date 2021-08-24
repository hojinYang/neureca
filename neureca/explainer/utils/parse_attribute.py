"""
Parse attributes
"""
import pickle
from pathlib import Path
from typing import Dict, List, Union
import re
import pandas as pd
import yaml
from tqdm import tqdm
from transformers import pipeline
from spacy.lang.en import English


class Attribute:
    """
    class Attribute
    """

    def __init__(self, d: Dict[str, str]):
        self.name = d["attr"]
        self.syn = d.get("syn", [])
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
        self.nlp = English()

        self.nlp.add_pipe("sentencizer")

    def generate_attribute(self, attribute_path: Path) -> None:
        with open(attribute_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        st = data["attribute"]

        while len(st) > 0:
            attr = st.pop()
            if attr.get("is_item") is True:
                continue

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

    def _read_single_review(self, review) -> Dict[str, List[Union[int, float, str]]]:
        review_dict: Dict[str, List[Union[int, float, str]]] = {
            "user": list(),
            "item": list(),
            "review_line": list(),
            "attr": list(),
            "sentiment": list(),
        }

        # user, item = review_text_path.stem.split("-")
        lines = self.nlp(review.review_text)

        for line in lines.sents:

            line = re.sub(";", "", line.text.lower().strip())
            review_dict["attr"] = list(self._read_single_review_line(line))
            review_dict["review_line"] = [line] * len(review_dict["attr"])

        review_dict["user"] = [review.user] * len(review_dict["attr"])
        review_dict["item"] = [review.item] * len(review_dict["attr"])

        if len(review_dict["review_line"]) > 0:
            sents = self.classifier(review_dict["review_line"])
            review_dict["sentiment"] = [
                x["score"] if x["label"] == "POSITIVE" else -x["score"] for x in sents
            ]

        return review_dict

    def build_dataset(self, review_path: Path) -> None:
        review_dict: Dict[str, List[Union[int, float, str]]] = {
            "user": list(),
            "item": list(),
            "review_line": list(),
            "attr": list(),
            "sentiment": list(),
        }

        reviews = pd.read_csv(
            review_path,
            header=0,
            usecols=[0, 1, 3],
            names=["user", "item", "review_text"],
            engine="python",
        )
        print(reviews)

        for review in tqdm(reviews.itertuples(index=True, name="Pandas"), total=len(reviews)):
            _review_dict = self._read_single_review(review)
            for key in review_dict:
                review_dict[key] += _review_dict[key]

        return pd.DataFrame(review_dict)


def save_db_and_attr_list(review_path, attribute_path, db_path, attribute_save_path):
    ap = AttributeParser()
    ap.generate_attribute(attribute_path)
    db = ap.build_dataset(review_path)
    db.to_csv(str(db_path), index=False)
    with open(str(attribute_save_path), "wb") as f:
        pickle.dump(ap.attributes, f)