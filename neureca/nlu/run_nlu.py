import pickle
from pathlib import Path
from neureca.nlu.training.run_experiment import _import_class

TRAIN_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "demo-toronto" / "preprocessed" / "train.pkl"
)


def load_model(dir_args_and_checkpoint: Path):

    with open(str(dir_args_and_checkpoint / "args.pkl"), "rb") as f:
        args = pickle.load(f)

    data_class = _import_class(f"neureca.nlu.data.{args.data_class}")
    model_class = _import_class(f"neureca.nlu.{args.model_type}.{args.model_class}")
    feat_class = _import_class(f"neureca.nlu.featurizers.{args.featurizer_class}")
    featurizer = feat_class(args)
    data = data_class(featurizer=featurizer, args=args)

    ckpt_path = list(dir_args_and_checkpoint.glob("*.ckpt"))[0]
    model = model_class.load_from_checkpoint(ckpt_path, data_config=data.config(), args=args)

    model.eval()
    model.freeze()

    del data_class, feat_class

    return model


class NLU:
    def __init__(self):
        self.intent_model = load_model(INTENT_PATH)
        with open(str(TRAIN_DATA_PATH), "rb") as f:
            self.data = pickle.load(f)


if __name__ == "__main__":
    print(TRAIN_DATA_PATH)
    # with
    path = Path("/home/hojin/code/neureca/neureca/nlu/training/logs/default/version_5/checkpoints")
    model = load_model(path)
