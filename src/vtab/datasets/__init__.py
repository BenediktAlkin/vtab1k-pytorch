def create_dataset(dataset, **kwargs):
    from vtab.utils.factory import instantiate
    kind = dataset.pop("kind")
    return instantiate(
        module_name="vtab.datasets",
        type_name=kind,
        **dataset,
        **kwargs,
    )
