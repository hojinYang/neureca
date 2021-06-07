import importlib


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'neureca.nlu.classifiers.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)  # [neureca.nlu.classifiers, MLP]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_
