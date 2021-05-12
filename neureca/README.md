
The directory structure of `neureca` is organized as follows:
```
.
├── app
│   └── neureca_api.py
├── nlu
│   ├── nlu.py
│   ├── featurizers
│   ├── classifiers
│   ├── recognizers
│   ├── data
│   └── training
├── recommender
│   ├── recommender.py
│   ├── cf_models
│   ├── data
│   └── training
├── explanator
│   ├── explanator.py
│   └── utils
├── dialogue_manager
│   ├── box.py
│   ├── bubble.py
│   └── manager.py
└── README.md

```

Within `neureca`,  there is breakdown between `nlu`, `recommender`, `explainer`, `dialogue_manager` and `app`. 

- `nlu` module is responsible for understanding natural language. Specifically, in this module we will train a neural networks for intent classifier and attribute recognizer. Raw text from user would be converted the into embedding using our featurizers which neural networks takes as input for training. 
- `recommender` module is responsible for generating recommendations for user. 
- In `explainer` module, we generate explanations or answers for the user's question based on review data. 
- `dialogue_manager` contains basic components developers needs when designing dialogue policy
- `app` contains NeurecaApi, a RESTful API that allows deveopers to deploy a Neureca model as an application.

In `nlu` and `recommender`, there is further breakdown between `data`, models(e.g. `recognizers`, `classifiers`, and `cf_models`) and `training`. 

### Data

There are three scopes of our code dealing with data, with slightly overlapping names: `DataModule`, `DataLoader`, and `Dataset`.

At the top level are `DataModule` classes, which are responsible for quite a few things:

- Downloading raw data and/or generating synthetic data
- Processing data as needed to get it ready to go through PyTorch models
- Splitting data into train/val/test sets
- Specifying dimensions of the inputs (e.g. `(C, H, W) float tensor`
- Specifying information about the targets (e.g. a class mapping)
- Specifying data augmentation transforms to apply in training

In the process of doing the above, `DataModule`s make use of a couple of other classes:

1. They wrap underlying data in a `torch Dataset`, which returns individual (and optionally, transformed) data instances.
2. They wrap the `torch Dataset` in a `torch DataLoader`, which samples batches, shuffles their order, and delivers them to the GPU.

If need be, you can read more about these [PyTorch data interfaces](https://pytorch.org/docs/stable/data.html).


### Models
Models are what is commonly known as "neural nets": code that accepts an input, processes it through layers of computations, and produces an output.

We use PyTorch-Lightning for training, which defines the `LightningModule` interface that handles not only everything that a Model handles, but also specifies the details of the learning algorithm: what loss should be computed from the output of the model and the ground truth, which optimizer should be used, with what learning rate, etc.

### Training

Our `training/run_experiment.py` is a script that handles many command-line parameters.

For example, Here's a command we can run to train intent classifier:

```sh
python neureca/nlu/training/run_experiment.py  --data_class Intent --featurizer_class Bert --model_type classifiers --model_class MLP
```
