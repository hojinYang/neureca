import argparse
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import numpy as np

MAX_LENGTH = 50
BERT_DIM = 768


class Bert:
    def __init__(self, args: argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}
        self.is_sequence = self.args.get("sequence", False)
        self.max_length = self.args.get("max_length", MAX_LENGTH)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model.eval()
        if self.is_sequence:
            self.feature_dims = (self.max_length, BERT_DIM)
        else:
            self.feature_dims = (BERT_DIM,)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--sequence", type=bool, default=False, help="avg emb vs seq emb")
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
        return parser

    def featurize(self, sentence):
        if self.is_sequence:

            with torch.no_grad():
                inputs = self.tokenizer(
                    sentence,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    return_offsets_mapping=True,
                )
                outputs = self.model(inputs["input_ids"])

                features = outputs[0].squeeze().numpy()
                mask = inputs["attention_mask"].squeeze().type(torch.ByteTensor).numpy()
                offset_mapping = inputs["offset_mapping"].squeeze().numpy()
            ret = {"features": features, "mask": mask, "offset_mapping": offset_mapping}

        else:

            with torch.no_grad():
                inputs = self.tokenizer(sentence, return_tensors="pt")
                # See the models docstrings for the detail of the inputs
                outputs = self.model(**inputs)
                # Transformers models always output tuples.
                # In our case, the first element is the hidden state of the last layer of the Bert model
                features = outputs[0].squeeze()[0].numpy()
            ret = {"features": features}

        return ret

    def get_bio_tags(self, attributes, attribute_mapper, offset_mapping):

        bio_tags = list()

        for offset in offset_mapping:
            bio = len(attribute_mapper) * 2

            if offset[1] == 0:
                bio_tags.append(bio)
                continue

            for attr in attributes:
                if offset[0] == attr["start_idx"] and offset[1] <= attr["end_idx"]:
                    bio = attribute_mapper[attr["attr_type"]]
                    break

                if offset[0] > attr["start_idx"] and offset[1] <= attr["end_idx"]:
                    bio = attribute_mapper[attr["attr_type"]] + len(attribute_mapper)
                    break

            bio_tags.append(bio)

        bio_tags = np.array(bio_tags)

        return bio_tags

    def get_attributes(self, uttr, bio_tags, offsets, attr_list):
        ret = dict()
        i = 0

        while i < len(bio_tags):
            bio = bio_tags[i]
            if bio < len(attr_list):
                start, end = offsets[i]
                temp = uttr[start:end]
                while i + 1 < len(bio_tags) and bio_tags[i + 1] == bio + len(attr_list):
                    i = i + 1
                    start, end = offsets[i]
                    temp = temp + uttr[start:end]

                attr = attr_list[bio]
                if attr not in ret:
                    ret[attr] = list()
                ret[attr].append(temp)

            i = i + 1

        return ret
