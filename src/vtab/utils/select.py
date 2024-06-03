def select(obj, path):
    if path is not None:
        for p in path.split("."):
            if isinstance(obj, dict):
                obj = obj[p]
            elif isinstance(obj, list):
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
    return obj

def select_set(obj, path, value):
    """
    selects a property from object given a path and assigns the value to it
    Example:
         obj = dict(dataset=dict(kind="vtab1k_dataset"))
         obj2 = select_set(obj, path="dataset.version", value="cifar")
         assert obj2 == dict(dataset=dict(kind="vtab1k_dataset", version="cifar"))
    """
    # split off last path item
    split = path.split(".")
    key = split[-1]
    path = ".".join(split[:-1])
    # select
    obj = select(obj=obj, path=path)
    # set value
    if isinstance(obj, dict):
        obj[key] = value
    else:
        raise NotImplementedError
    return obj