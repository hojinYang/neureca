import argparse
from typing import Dict, Any
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
from .base_featurizer import BaseFeaturizer

BERT_DIM = 768


class Bert(BaseFeaturizer):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args=args)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.model.eval()
        self.feature_dims = (self.max_length, BERT_DIM)

    def featurize(self, sentence: str, use_sentence_emb: bool) -> Dict[str, Any]:
        with torch.no_grad():
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                padding="max_length",
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            bert_outputs = self.model(inputs["input_ids"])["last_hidden_state"]
            features = bert_outputs.squeeze().numpy()

            if use_sentence_emb:
                features = features[0]
                outputs = {"features": features}

            else:
                mask = inputs["attention_mask"].squeeze().type(torch.ByteTensor).numpy()
                offset_mapping = inputs["offset_mapping"].squeeze().numpy()
                outputs = {"features": features, "mask": mask, "offset_mapping": offset_mapping}

        return outputs
