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
    """Import class from a module, e.g. 'neureca.recommender.cf_models.AE'"""
    module_name, class_name = module_and_class_name.rsplit(
        ".", 1
    )  # [neureca.recommender.cf_models, AE]
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

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f"neureca.recommender.data.{temp_args.data_class}")
    model_class = _import_class(
        f"neureca.recommender.{temp_args.model_type}.{temp_args.model_class}"
    )

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    parser.add_argument("--help", "-h", action="help")

    return parser


def main():
    """
    Run an experiment
    sample command
    '''
    python neureca/recommender/training/run_experiment.py  --data_class UserBased --model_type cf_models --model_class AE
    """

    parser = _setup_parser()
    args = parser.parse_args()

    data_class = _import_class(f"neureca.recommender.data.{args.data_class}")
    model_class = _import_class(f"neureca.recommender.{args.model_type}.{args.model_class}")
    data = data_class(args=args)

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
    logger = pl.loggers.TensorBoardLogger("neureca/recommender/training/logs")

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        weights_save_path=Path(__file__).resolve().parents[0] / "logs" / str(args.data_class),
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
