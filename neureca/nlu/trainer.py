"""
Experiment running framework.
Refer to https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs
"""
import argparse
import pickle
from pathlib import Path
import pytorch_lightning as pl
from neureca.shared.utils import import_class


def _setup_parser():
    """
    parse argument
    """
    parser = argparse.ArgumentParser(add_help=False)
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    parser.add_argument("--data_class", type=str)
    parser.add_argument("--model_class", type=str)
    parser.add_argument("--lit_wrapper_class", type=str)
    parser.add_argument("--featurizer_class", type=str)
    parser.add_argument("--demo_path", type=str)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"neureca.nlu.data.{temp_args.data_class}")
    model_class = import_class(f"neureca.shared.models.{temp_args.model_class}")
    lit_wrapper_class = import_class(f"neureca.nlu.lit_wrappers.{temp_args.lit_wrapper_class}")
    feat_class = import_class(f"neureca.nlu.featurizers.{temp_args.featurizer_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    feat_group = parser.add_argument_group("Feat Args")
    feat_class.add_to_argparse(feat_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_wrapper_group = parser.add_argument_group("Lit Wrapper Args")
    lit_wrapper_class.add_to_argparse(lit_wrapper_group)

    parser.add_argument("--help", "-h", action="help")

    return parser


def main():
    """
    Run an experiment
    sample command
    '''
    python neureca/nlu/trainer.py  --data_class Intent --featurizer_class Bert --model_class MLP --lit_wrapper_class Classifier --path /home/hojin/code/neureca/demo-toronto --use_sentence_emb True
    python neureca/nlu/trainer.py  --data_class Attribute --featurizer_class Bert --model_class LSTM --lit_wrapper_class CRFRecognizer --path /home/hojin/code/neureca/demo-toronto
    python cmd.py  --data_class Intent --featurizer_class Bert --model_class MLP --lit_wrapper_class Classifier --path /home/hojin/code/neureca/demo-toronto --use_sentence_emb True
    """

    parser = _setup_parser()
    args = parser.parse_args()

    data_class = import_class(f"neureca.nlu.data.{args.data_class}")
    model_class = import_class(f"neureca.shared.models.{args.model_class}")
    lit_wrapper_class = import_class(f"neureca.nlu.lit_wrappers.{args.lit_wrapper_class}")
    feat_class = import_class(f"neureca.nlu.featurizers.{args.featurizer_class}")

    featurizer = feat_class(args)
    data = data_class(featurizer=featurizer, args=args)
    model = model_class(data_config=data.config(), args=args)

    if args.load_checkpoint is not None:
        lit_wrapper = lit_wrapper_class(args.load_checkpoint, model=model, args=args)
    else:
        lit_wrapper = lit_wrapper_class(model=model, args=args)

    val_metric = lit_wrapper.get_main_validation_metric()

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor=val_metric["name"], mode=val_metric["mode"], patience=10
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch:03d}-{val_loss:.3f}",
        monitor=val_metric["name"],
        mode=val_metric["mode"],
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]
    log_path = Path(args.path) / "weights"
    logger = pl.loggers.TensorBoardLogger(log_path, name=str(args.data_class))

    args.weights_summary = "full"  # Print full summary of the model
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        weights_save_path=log_path,
    )
    # pylint: disable=no-member
    trainer.tune(lit_wrapper, datamodule=data)
    trainer.fit(lit_wrapper, datamodule=data)
    trainer.test(lit_wrapper, datamodule=data)
    # pylint: enable=no-member

    best_model_path = model_checkpoint_callback.best_model_path

    if best_model_path:
        print("Best model saved at:", best_model_path)
        with open(Path(best_model_path).resolve().parents[1] / "args.pkl", "wb") as f:
            pickle.dump(args, f)


if __name__ == "__main__":
    main()
