from itertools import product

import wandb
import yaml

from vtab.utils.formatting import dict_to_string
from .base_processor import BaseProcessor


class ResolveGridsearchProcessor(BaseProcessor):
    """ generates all tuning runs from a given config """

    def run(self):
        self.logger.info(f"generating '{self.pipeline_uri.as_posix()}' runs '{self.work_area.root.as_posix()}'")
        with open(self.pipeline_uri) as f:
            pipeline_config = yaml.safe_load(f)
        # generate pipeline_id and add it on top
        if "pipeline_id" not in pipeline_config:
            pipeline_id = wandb.util.generate_id()
            pipeline_config = dict(pipeline_id=pipeline_id, **pipeline_config)
        else:
            pipeline_id = pipeline_config["pipeline_id"]

        # consume current_stage
        current_stage_name = pipeline_config["current_stage"]
        current_stage_config = pipeline_config[current_stage_name]
        pipeline_config.pop(current_stage_name)
        next_stage_name = current_stage_config["next_stage"]
        pipeline_config["current_stage"] = next_stage_name

        # check current_stage
        assert len(current_stage_config) == 2
        assert "next_stage" in current_stage_config
        assert "processor" in current_stage_config

        # get grid of next stage
        next_stage_config = pipeline_config[next_stage_name]
        grid = next_stage_config.pop("grid")
        self.logger.info("grid:")
        for key, value in grid.items():
            self.logger.info(f"- {key}: {value}")

        # extract static_params (these hyperparameters only have 1 possibility -> they dont change)
        static_params = {}
        for key, value in list(grid.items()):
            if isinstance(value, list):
                if len(value) == 1:
                    # hyperparameter is list with 1 possibility -> no search
                    static_params[key] = value[0]
                    grid.pop(key)
            else:
                # hyperparameter is a scalar -> no search
                static_params[key] = value
                grid.pop(key)
        # log
        self.logger.info("-" * 50)
        self.logger.info("static parameters:")
        for key, value in static_params.items():
            self.logger.info(f"- {key}: {value}")

        # create dynamic_params (parameters that are varied)
        dynamic_params = []
        keys = grid.keys()
        for instance in product(*grid.values()):
            dynamic_params.append(dict(zip(keys, instance)))
        # print
        self.logger.info("-" * 50)
        self.logger.info(f"dynamic parameters (len={len(dynamic_params)}):")
        for key, value in grid.items():
            self.logger.info(f"- {key}: {value}")

        # store to work_area
        prefix = f"{self.pipeline_uri.name.replace('.yaml', '')}__{pipeline_id}__{next_stage_name}"
        for i, dynamic_param in enumerate(dynamic_params):
            if len(dynamic_param) > 0:
                for value in dynamic_param.values():
                    if isinstance(value, dict):
                        raise NotImplementedError("dict in dynamic_params not supported")
                run_pipeline_uri = self.work_area.root / f"{prefix}__{dict_to_string(dynamic_param)}.yaml"
            else:
                run_pipeline_uri = self.work_area.root / f"{prefix}.yaml"
            self.logger.info(f"creating run config {i + 1}/{len(dynamic_params)} ({run_pipeline_uri.as_posix()})")

            next_stage_config["hyperparams"] = dict(**static_params, **dynamic_param)
            next_stage_config["num_runs"] = len(dynamic_params)
            with open(run_pipeline_uri, "w") as f:
                yaml.safe_dump(pipeline_config, f, sort_keys=False)
