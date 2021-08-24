import sys
from neureca.nlu.trainer import main as nlu_main
from neureca.recommender.trainer import main as rec_main


def neureca_train_command():
    if len(sys.argv) < 2:
        raise ValueError("you must specify the mode to train model")
    mode = sys.argv.pop(1)
    if mode not in ["intent", "attribute", "recommender"]:
        raise ValueError("first argument should be intent, attribute or recommender")

    if mode == "intent":
        sys.argv += [
            "--data_class",
            "Intent",
            "--featurizer_class",
            "Bert",
            "--model_class",
            "MLP",
            "--lit_wrapper_class",
            "Classifier",
        ]
        nlu_main()

    elif mode == "attribute":
        sys.argv += [
            "--data_class",
            "Attribute",
            "--featurizer_class",
            "Bert",
            "--model_class",
            "LSTM",
            "--lit_wrapper_class",
            "CRFRecognizer",
        ]
        nlu_main()
    elif mode == "recommender":
        sys.argv += [
            "--data_class",
            "UserBased",
            "--model_class",
            "AutoRec",
            "--lit_wrapper_class",
            "ExplicitRatingWrapper",
        ]
        rec_main()


if __name__ == "__main__":
    neureca_train_command()
