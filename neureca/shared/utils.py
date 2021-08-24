from pathlib import Path
import importlib
import os


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'neureca.nlu.classifiers.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)  # [neureca.nlu.classifiers, MLP]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def find_latest_subdir(path: Path):
    all_subdirs = [d for d in path.iterdir() if d.is_dir()]
    latest_subdir = max(all_subdirs, key=os.path.getmtime)
    return latest_subdir
