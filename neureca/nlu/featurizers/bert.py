import argparse
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
import numpy as np

MAX_LENGTH = 50
BERT_DIM = 768


class Bert:
    def __init__(self, args: argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}
        self.is_sequence = isinstance(self.args.get("sequence", None), (str, int))
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
        parser.add_argument("--is_sequence", type=bool, default=False, help="avg emb vs seq emb")
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
        return parser

    def featurize(self, sentence):
        if self.is_sequence:
            with torch.no_grad():
                inputs = self.tokenizer(
                    sentence, return_tensors="pt", padding="max_length", max_length=self.max_length
                )
                outputs = self.model(**inputs)
                features = outputs[0].squeeze().numpy()

        else:
            inputs = self.tokenizer(sentence, return_tensors="pt")
            with torch.no_grad():
                # See the models docstrings for the detail of the inputs
                outputs = self.model(**inputs)
                # Transformers models always output tuples.
                # In our case, the first element is the hidden state of the last layer of the Bert model
                features = outputs[0].squeeze()[0].numpy()

        return features

    def get_bio_tags(self, sentence, attributes, attribute_mapper):
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
            mask = inputs["attention_mask"].squeeze().numpy()
            offset_mapping = inputs["offset_mapping"].squeeze().numpy()

        # return_offsets_mapping

        bio_tags = list()

        for offset in offset_mapping:
            bio = len(attribute_mapper) * 2 + 1

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

        print(sentence)
        print(offset_mapping)
        print(bio_tags)
        print(mask)
        print(features.shape, mask.shape, bio_tags.shape)

        return features, mask, bio_tags
