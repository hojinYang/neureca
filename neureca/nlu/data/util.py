"""
BaseDataset class
"""
from typing import Sequence, Union, Any, Callable, Tuple, Dict
from pathlib import Path
import random
import yaml
import re
import torch


SequenceOrTensor = Union[Sequence, torch.tensor]


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.
    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if len(targets) != len(data):
            raise ValueError("Data and targets must be of equal length")

        self.data = data
        self.targets = targets
        self.data_transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return length of the dataset"""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by trasform function
        """
        datum, target = self.data[index], self.targets[index]

        if self.data_transform is not None:
            datum = self.data_transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target


class BaseDatasetWithMask(torch.utils.data.Dataset):
    """
    Base Dataset class that simply processes data and targets through optional transforms.
    Parameters
    ----------
    data
        commonly these are torch tensors, numpy arrays, or PIL Images
    targets
        commonly these are torch tensors or numpy arrays
    transform
        function that takes a datum and returns the same
    target_transform
        function that takes a target and returns the same
    """

    def __init__(
        self,
        data: SequenceOrTensor,
        targets: SequenceOrTensor,
        masks: SequenceOrTensor,
        transform: Callable = None,
        target_transform: Callable = None,
    ) -> None:
        if (
            (len(targets) != len(data))
            or (len(targets) != len(masks))
            or (len(targets) != len(masks))
        ):
            raise ValueError("Data and targets must be of equal length")

        self.data = data
        self.targets = targets
        self.masks = masks
        self.data_transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Return length of the dataset"""
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Return a datum and its target, after processing by trasform function
        """
        datum, target, mask = self.data[index], self.targets[index], self.masks[index]

        if self.data_transform is not None:
            datum = self.data_transform(datum)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return datum, target, mask


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
    def __init__(self, nlu_path, attr_path):
        self.nlu_path: Path = nlu_path
        self.attr_path: Path = attr_path
        self.attr_dict: Dict[str, Attribute] = dict()

    def update_attribute_dict(self) -> None:

        with open(self.attr_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        for attr in data["attribute"]:
            _ = self._rec(attr)

    def _rec(self, attr: Dict[str, str]) -> Attribute:

        if "sub-attr" not in attr:
            attr_cls = Attribute(attr)

        else:
            children = [self._rec(child) for child in attr["sub-attr"]]
            attr_cls = Attribute(attr)
            attr_cls.children = children

        self.attr_dict[attr_cls.name] = attr_cls
        return attr_cls

    def convert(self):
        output = list()

        intents, attributes = list(), list()

        with open(self.attr_path) as f:
            data_attr = yaml.load(f, Loader=yaml.FullLoader)

        for attr in data_attr["attribute"]:
            attributes.append(attr["attr"])

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
            "attributes": attributes,
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


if __name__ == "__main__":
    p_attr = Path("/home/hojin/code/neureca/demo-toronto/data/attribute.yaml")
    p_nlu = Path("/home/hojin/code/neureca/demo-toronto/data/nlu.yaml")
    converter = NLUYamlToTrainConverter(p_nlu, p_attr)
    converter.update_attribute_dict()

    print(converter.convert())
