import argparse
from pathlib import Path
import pickle

from neureca.shared.data import BaseDataModule
from neureca.nlu.data.util import convert_yaml_to_training_data

NLU_CONVERTED_FILENAME = "nlu_converted.pkl"


class BaseNLUDataModule(BaseDataModule):
    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        nlu_converted_filename = Path(
            self.args.get("nlu_converted_filename", NLU_CONVERTED_FILENAME)
        )
        self.nlu_converted_data_path = self.prepocessed_dirname / nlu_converted_filename
        self.num_intents: int
        self.num_attribute_tags: int
        self.prepare_data()

    def prepare_data(self):
        if self.nlu_converted_data_path.exists():
            with open(str(self.nlu_converted_data_path), "rb") as f:
                nlu_converted_data = pickle.load(f)
                self.num_intents = len(nlu_converted_data["intents"])
                self.num_attribute_tags = len(nlu_converted_data["attribute_tags"])
            return

        nlu_converted_data = convert_yaml_to_training_data(
            self.nlu_path, self.attribute_path, self.rating_path
        )
        self.num_intents = len(nlu_converted_data["intents"])
        self.num_attribute_tags = len(nlu_converted_data["attribute_tags"])

        with open(str(self.nlu_converted_data_path), "wb") as f:
            pickle.dump(nlu_converted_data, f)
