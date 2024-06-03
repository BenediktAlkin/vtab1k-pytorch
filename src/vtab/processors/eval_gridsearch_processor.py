import shutil

import pandas as pd
import yaml

from vtab.utils.naming import (
    change_fname,
    get_pipeline_fnames,
    fname_to_dynamic_params,
    fname_to_paramstr,
    remove_seed_from_paramstr,
)
from .base_processor import BaseProcessor


class EvalGridsearchProcessor(BaseProcessor):
    """ selects best hyperparameters from tuning runs and generates eval yamls """

    def __init__(self, seeds, train_split, eval_split, **kwargs):
        super().__init__(**kwargs)
        self.seeds = seeds
        self.train_split = train_split
        self.eval_split = eval_split

    def run(self):
        with open(self.pipeline_uri) as f:
            pipeline_config = yaml.safe_load(f)
        stage_config = pipeline_config[pipeline_config["current_stage"]]
        pipeline_id = pipeline_config["pipeline_id"]

        # fetch waiting files
        waiting_fnames = get_pipeline_fnames(
            uri=self.work_area.waiting_area,
            pipeline_id=pipeline_id,
        )
        if len(waiting_fnames) == 0:
            return
        # fetch num_runs from first yaml
        with open(self.work_area.waiting_area / waiting_fnames[0]) as f:
            waiting_config = yaml.safe_load(f)
        num_runs = waiting_config[waiting_config["current_stage"]]["num_runs"]
        if len(waiting_fnames) < num_runs:
            return
        assert len(waiting_fnames) == num_runs

        # read all results
        tune_results = []
        for fname in waiting_fnames:
            with open(self.work_area.waiting_area / fname) as f:
                result = yaml.safe_load(f)["result"]
            tune_results.append(
                dict(
                    fname=fname,
                    result=result,
                    **fname_to_dynamic_params(fname),
                ),
            )
        # find best average result
        df = pd.DataFrame(tune_results)
        group_by = [key for key in tune_results[0].keys() if key not in ["fname", "result", "seed"]]
        df = df.groupby(group_by).agg({"fname": "first", "result": "mean"}).reset_index()
        best_idx = df["result"].idxmax()
        best_mean_result = df.loc[best_idx]["result"]
        best_fname = df.loc[best_idx]["fname"]
        best_paramstr = remove_seed_from_paramstr(fname_to_paramstr(best_fname))
        self.logger.info(f"best accuracy over seeds: {best_mean_result} ({best_paramstr})")

        with open(self.work_area.waiting_area / best_fname) as f:
            best_config = yaml.safe_load(f)
        # remove old seed + change dataset splits
        hyperparams = best_config[best_config["current_stage"]]["hyperparams"]
        hyperparams.pop("seed")
        hyperparams["train_dataset"]["split"] = self.train_split
        hyperparams["eval_dataset"]["split"] = self.eval_split

        # update pipeline
        next_stage = stage_config["next_stage"]
        pipeline_config.pop(pipeline_config["current_stage"])
        pipeline_config["current_stage"] = next_stage
        pipeline_config[next_stage]["hyperparams"] = hyperparams
        pipeline_config[next_stage]["num_runs"] = len(self.seeds)

        # generate yamls
        eval_base_fname = change_fname(fname=best_fname, stage_name=next_stage, paramstr=best_paramstr)
        for seed in self.seeds:
            hyperparams["seed"] = seed
            with open(self.work_area.root / f"{eval_base_fname}_seed={seed}.yaml", "w") as f:
                yaml.safe_dump(pipeline_config, f, sort_keys=False)

        # move yamls from waiting -> finished
        for waiting_fname in waiting_fnames:
            shutil.move(self.work_area.waiting_area / waiting_fname, self.work_area.finished_area)
