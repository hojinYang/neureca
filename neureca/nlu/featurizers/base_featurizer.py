from abc import ABC, abstractmethod
from typing import Dict, Any

MAX_LENGTH = 50


class BaseFeaturizer(ABC):
    @abstractmethod
    def featurize(self, sentence: str, use_sentence_emb: bool) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
        return parser