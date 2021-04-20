"""
Experiment running framework.
Refer to https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs
"""
import argparse
import importlib
import pickle
from pathlib import Path

import pytorch_lightning as pl


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'neureca.nlu.classifiers.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)  # [neureca.nlu.classifiers, MLP]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def _setup_parser():
    """
    parse argument
    """
    parser = argparse.ArgumentParser(add_help=False)
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    parser.add_argument("--data_class", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--model_class", type=str)
    parser.add_argument("--featurizer_class", type=str)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"neureca.nlu.data.{temp_args.data_class}")
    model_class = _import_class(f"neureca.nlu.{temp_args.model_type}.{temp_args.model_class}")
    feat_class = _import_class(f"neureca.nlu.featurizers.{temp_args.featurizer_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    feat_group = parser.add_argument_group("Feat Args")
    feat_class.add_to_argparse(feat_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser


def main():
    """
    Run an experiment
    sample command
    '''
    python neureca/nlu/training/run_experiment.py  --data_class Intent --featurizer_class Bert --model_type classifiers --model_class MLP
    """

    parser = _setup_parser()
    args = parser.parse_args()

    data_class = _import_class(f"neureca.nlu.data.{args.data_class}")
    model_class = _import_class(f"neureca.nlu.{args.model_type}.{args.model_class}")
    feat_class = _import_class(f"neureca.nlu.featurizers.{args.featurizer_class}")
    featurizer = feat_class(args)
    data = data_class(featurizer=featurizer, args=args)

    if args.load_checkpoint is not None:
        model = model_class.load_from_checkpoint(
            args.load_checkpoint, data_config=data.config(), args=args
        )
    else:
        model = model_class(data_config=data.config(), args=args)

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}-{valid_acc_epoch:.3f}", monitor="val_loss", mode="min"
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]
    logger = pl.loggers.TensorBoardLogger("neureca/nlu/training/logs")

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        weights_save_path=Path(__file__).resolve().parents[0] / "logs",
    )
    # pylint: disable=no-member
    trainer.tune(model, datamodule=data)
    trainer.fit(model, datamodule=data)
    trainer.test(model, datamodule=data)
    # pylint: enable=no-member

    best_model_path = model_checkpoint_callback.best_model_path
    if best_model_path:
        print("Best model saved at:", best_model_path)
        print(best_model_path)
        print(Path(best_model_path).resolve().parents[0])
        with open(Path(best_model_path).resolve().parents[0] / "args.pkl", "wb") as f:
            pickle.dump(args, f)


if __name__ == "__main__":
    main()
