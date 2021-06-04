"""
BaseDataset class
"""
from typing import Sequence, Union, Any, Callable, Tuple, Dict
from pathlib import Path
import random
import yaml
import re
import torch
import pandas as pd
import numpy as np


SequenceOrTensor = Union[Sequence, torch.tensor]


class Attribute:
    """
    class Attribute
    """

    def __init__(self, d: Dict[str, str]):
        self.name = d["attr"]
        self.syn = d.get("syn", None)
        self.children = list()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class PatternMatcher:
    def __init__(self, attr_string, attr_type, attr_dict):
        self.attr_string = attr_string
        self.attr_type = attr_type
        self.attr_list = list()
        self.original_phrase = "[" + attr_string + "]" + "{" + attr_type + "}"

        self._cursor = 0
        self.generate_attr_list(attr_dict)

    def generate_attr_list(self, attr_dict):
        target = attr_dict[self.attr_type]

        if "*" not in self.attr_string:
            attr_list = self.attr_string.split("/")
            for attr in attr_list:
                self.attr_list.append(attr_dict[attr])
        else:
            for c in target.children:
                self.attr_list.append(attr_dict[c.name])

        random.shuffle(self.attr_list)

    def run(self, uttr):

        cur_attr = self.attr_list[self._cursor]
        replace_words = random.sample(cur_attr.syn, 1)[0]

        start_idx = uttr.find(self.original_phrase)
        end_idx = start_idx + len(replace_words)
        attr_type = {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "text": replace_words,
            "attr_type": self.attr_type,
        }

        replaced_uttr = uttr.replace(self.original_phrase, replace_words, 1)
        self._cursor += 1

        if self._cursor >= len(self.attr_list):
            self._cursor = 0
            random.shuffle(self.attr_list)

        return replaced_uttr, attr_type


class NLUYamlToTrainConverter:
    def __init__(self, nlu_path, attr_path, rating_path):
        self.nlu_path: Path = nlu_path
        self.attr_path: Path = attr_path
        self.rating_path: Path = rating_path
        self.attr_dict: Dict[str, Attribute] = dict()

    def update_attribute_dict(self) -> None:

        with open(self.attr_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        for attr in data["attribute"]:
            _ = self._rec(attr)

    def _rec(self, attr: Dict[str, str]) -> Attribute:

        if "is_item" in attr:
            rating = pd.read_csv(str(self.rating_path))

            item_names = rating["name"].unique().tolist()
            attr["syn"] = item_names

            attr_cls = Attribute(attr)

        elif "sub-attr" not in attr:
            attr_cls = Attribute(attr)

        else:
            children = [self._rec(child) for child in attr["sub-attr"]]
            attr_cls = Attribute(attr)
            attr_cls.children = children

        self.attr_dict[attr_cls.name] = attr_cls
        return attr_cls

    def convert(self):
        output = list()

        intents, attribute_tags = list(), list()

        with open(self.attr_path) as f:
            data_attr = yaml.load(f, Loader=yaml.FullLoader)

        for attr in data_attr["attribute"]:
            attribute_tags += ["B-" + attr["attr"], "I-" + attr["attr"]]
        attribute_tags.append("O")

        preferences = list(self.attr_dict.keys())

        with open(self.nlu_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        for i in data["nlu"]:
            intent = i["intent"]

            for example in i["examples"]:
                output += self._run(example, intent=intent)

            intents.append(intent)

        ret = {
            "intents": intents,
            "attribute_tags": attribute_tags,
            "preferences": preferences,
            "examples": output,
        }

        return ret

    def _run(self, text, intent, max_iter=10):
        preference = list()
        if "->" in text:
            uttr, pref = text.split("->")
            patturns = re.findall("\[(.*?)\]\{(.*?)\}", pref)
            for attr, attr_type in patturns:
                attr_list = attr.split("/")
                for a in attr_list:
                    d = {"attr": a, "attr_type": attr_type}
                    preference.append(d)
        else:
            uttr = text

        temp_max_iter = 5

        handlers = list()
        patturns = re.findall("\[(.*?)\]\{(.*?)\}", uttr)

        for attr, attr_type in patturns:
            pm = PatternMatcher(attr, attr_type, self.attr_dict)
            temp_max_iter = max(temp_max_iter, len(pm.attr_list))
            handlers.append(pm)

        max_iter = min(temp_max_iter, max_iter)

        ret = []
        for _ in range(max_iter):
            return_uttr = uttr
            attributes = list()
            for h in handlers:
                return_uttr, attr_type = h.run(return_uttr)
                attributes.append(attr_type)

            d = {
                "text": return_uttr.strip(),
                "intent": intent,
                "attributes": attributes,
                "preference": preference,
            }
            ret.append(d)

        return ret


def convert_yaml_to_training_data(nlu_file, attr_file, rating_file):
    converter = NLUYamlToTrainConverter(nlu_file, attr_file, rating_file)
    converter.update_attribute_dict()
    training_data = converter.convert()
    return training_data


def get_bio_tags(attributes, offsets, attribute_tag_mapper):

    bio_tags = list()

    for offset in offsets:
        tag = attribute_tag_mapper["O"]

        if offset[1] == 0:
            bio_tags.append(tag)
            continue

        for attr in attributes:
            if offset[0] == attr["start_idx"] and offset[1] <= attr["end_idx"]:
                tag = attribute_tag_mapper["B-" + attr["attr_type"]]
                break

            if offset[0] > attr["start_idx"] and offset[1] <= attr["end_idx"]:
                tag = attribute_tag_mapper["I-" + attr["attr_type"]]
                break

        bio_tags.append(tag)

    bio_tags = np.array(bio_tags)
    return bio_tags


def get_attributes(uttr, bio_tags, offsets, attr_list):
    ret = dict()
    i = 0

    while i < len(bio_tags):
        bio = bio_tags[i]
        if bio < len(attr_list):
            start, end = offsets[i]
            temp = uttr[start:end]
            while i + 1 < len(bio_tags) and bio_tags[i + 1] == bio + len(attr_list):
                prev_end = end
                i = i + 1
                start, end = offsets[i]
                temp = temp + (start - prev_end) * " " + uttr[start:end]

            attr = attr_list[bio]
            if attr not in ret:
                ret[attr] = list()
            ret[attr].append(temp)

        i = i + 1

    return ret


if __name__ == "__main__":
    p_attr = Path("/home/hojin/code/neureca/demo-toronto/data/attribute.yaml")
    p_nlu = Path("/home/hojin/code/neureca/demo-toronto/data/nlu.yaml")
    p_rating = Path("/home/hojin/code/neureca/demo-toronto/data/ratings.csv")
    converter = NLUYamlToTrainConverter(p_nlu, p_attr, p_rating)
    converter.update_attribute_dict()

    print(converter.convert())
