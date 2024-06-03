def create_model(model, **kwargs):
    from vtab.utils.factory import instantiate
    kind = model.pop("kind")
    return instantiate(
        module_name="vtab.models",
        type_name=kind,
        **model,
        **kwargs,
    )
