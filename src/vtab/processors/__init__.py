def create_processor(pipeline_uri, **kwargs):
    # load config
    import yaml
    from pathlib import Path
    pipeline_uri = Path(pipeline_uri).expanduser().with_suffix(".yaml")
    assert pipeline_uri.exists(), f"'{pipeline_uri.as_posix()}' doesn't exist"
    with open(pipeline_uri) as f:
        config = yaml.safe_load(f)

    # get stage
    stage_name = config["current_stage"]
    stage_config = config[stage_name]
    assert "processor" in stage_config, f"stage '{stage_name}' doesn't define a processor"
    processor = stage_config["processor"]

    # instantiate
    from vtab.utils.factory import instantiate
    kind = processor.pop("kind")
    return instantiate(
        module_name="vtab.processors",
        type_name=kind,
        pipeline_uri=pipeline_uri,
        **processor,
        **kwargs,
    )
