from abc import ABC, abstractmethod
from typing import Dict, Any
import argparse

MAX_LENGTH = 50


class BaseFeaturizer(ABC):
    def __init__(self, args: argparse.Namespace = None):
        self.args = vars(args) if args is not None else {}
        self.max_length = self.args.get("max_length", MAX_LENGTH)

    @abstractmethod
    def featurize(self, sentence: str, use_sentence_emb: bool) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
        return parser
