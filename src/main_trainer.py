import os
from argparse import ArgumentParser
from pathlib import Path

import kappaconfig as kc
import yaml

from vtab.trainers import create_trainer


def parse_args():
    parser = ArgumentParser(description="Trains a single model based on a pipeline_config file")
    parser.add_argument("--pipeline_config", type=str, required=True)
    parser.add_argument("--static_config", type=str, default="static_config.yaml")
    parser.add_argument("--devices", type=str, required=True)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--num_workers", type=int, default=10)
    return vars(parser.parse_args())


def main(pipeline_config, static_config, devices, accelerator, num_workers):
    # init device
    if "," in devices:
        raise NotImplementedError("multi-GPU not implemented")
    device = int(devices)

    # load static_config
    static_config = Path(static_config).expanduser().with_suffix(".yaml")
    static_config = kc.DefaultResolver().resolve(kc.from_file_uri(static_config))
    # set TORCH_HOME to download torchhub models into a specified directory instead of into HOME directory
    if "torch_home" in static_config:
        os.environ["TORCH_HOME"] = static_config["torch_home"]

    # load pipeline_config
    pipeline_uri = Path(pipeline_config).expanduser().with_suffix(".yaml")
    assert pipeline_uri.exists(), f"pipeline_config '{pipeline_uri.as_posix()}' doesn't exist"
    with open(pipeline_uri) as f:
        pipeline_config = yaml.safe_load(f)

    # create trainer
    stage = pipeline_config[pipeline_config["current_stage"]]
    hyperparams = stage["hyperparams"]
    trainer = create_trainer(
        hyperparams.pop("trainer"),
        static_config=static_config,
        device=device,
        accelerator=accelerator,
        num_workers=num_workers,
    )
    acc = trainer.train(hyperparams)
    # add accuracy to top of pipeline_config
    with open(pipeline_uri) as f:
        pipeline_config = yaml.safe_load(f)
    pipeline_config = dict(result=acc, **pipeline_config)
    with open(pipeline_uri, "w") as f:
        yaml.safe_dump(pipeline_config, f, sort_keys=False)
    print(f"accuracy: {acc:.4f}")


if __name__ == '__main__':
    main(**parse_args())
