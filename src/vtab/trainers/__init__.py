def create_trainer(trainer, **kwargs):
    from vtab.utils.factory import instantiate
    kind = trainer.pop("kind")
    return instantiate(
        module_name="vtab.trainers",
        type_name=kind,
        **trainer,
        **kwargs,
    )
