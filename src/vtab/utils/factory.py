import importlib


def instantiate(module_name: str, type_name: str, **kwargs):
    module = importlib.import_module(f"{module_name}.{type_name}")
    query = type_name.lower().replace("_", "")
    valid_type_names = list(filter(lambda k: k.lower() == query, module.__dict__.keys()))
    # filter out non-classes
    valid_type_names = [
        valid_type_name
        for valid_type_name in valid_type_names
        if isinstance(getattr(module, valid_type_name), type)
    ]
    assert len(valid_type_names) == 1
    ctor = getattr(module, valid_type_names[0])
    return ctor(**kwargs)
