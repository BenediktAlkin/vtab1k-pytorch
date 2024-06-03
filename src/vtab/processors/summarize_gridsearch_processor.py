import shutil

import numpy as np
import pandas as pd
import yaml

from vtab.utils.naming import fname_to_dynamic_params, get_pipeline_fnames, fname_to_summarized_name
from .base_processor import BaseProcessor


class SummarizeGridsearchProcessor(BaseProcessor):
    """
    summarizes a pipeline with the following steps:
    - train all hyperparameter combinations on the train set and evaluate on the validation set
    - select the best setting
    - train the best setting on the train + validation set and evaluate on the test set
    """

    def __init__(self, tune_stage_name, eval_stage_name, **kwargs):
        super().__init__(**kwargs)
        self.tune_stage_name = tune_stage_name
        self.eval_stage_name = eval_stage_name

    def run(self):
        with open(self.pipeline_uri) as f:
            pipeline_config = yaml.safe_load(f)
        pipeline_id = pipeline_config["pipeline_id"]

        # fetch eval files from waiting area
        eval_fnames = get_pipeline_fnames(
            uri=self.work_area.waiting_area,
            pipeline_id=pipeline_id,
            stage_name=self.eval_stage_name,
        )
        if len(eval_fnames) == 0:
            return
        # check num_runs by fetching num_runs from first yaml
        with open(self.work_area.waiting_area / eval_fnames[0]) as f:
            eval_config = yaml.safe_load(f)
        num_eval_runs = eval_config[eval_config["current_stage"]]["num_runs"]
        if len(eval_fnames) < num_eval_runs:
            return
        assert len(eval_fnames) == num_eval_runs

        # read all eval results (eval results are in waiting area)
        eval_results = []
        for fname in eval_fnames:
            with open(self.work_area.waiting_area / fname) as f:
                result = yaml.safe_load(f)["result"]
            eval_results.append(
                dict(
                    stage_name="eval",
                    result=result,
                    **fname_to_dynamic_params(fname),
                ),
            )
        # calculate mean_eval_result
        mean_eval_result = float(np.mean([eval_result["result"] for eval_result in eval_results]))

        # fetch all tune fnames (tune results are in finished area)
        tune_fnames = get_pipeline_fnames(
            uri=self.work_area.finished_area,
            pipeline_id=pipeline_id,
            stage_name=self.tune_stage_name,
        )
        # check num_runs by fetching num_runs from first yaml
        with open(self.work_area.finished_area / tune_fnames[0]) as f:
            tune_config = yaml.safe_load(f)
        num_tune_runs = tune_config[tune_config["current_stage"]]["num_runs"]
        assert len(tune_fnames) == num_tune_runs
        # read results
        tune_results = []
        for fname in tune_fnames:
            with open(self.work_area.finished_area / fname) as f:
                result = yaml.safe_load(f)["result"]
            tune_results.append(
                dict(
                    stage_name="tune",
                    result=result,
                    **fname_to_dynamic_params(fname),
                ),
            )
        # calculate aggregated metrics
        df = pd.DataFrame(tune_results)
        group_by = [key for key in tune_results[0].keys() if key not in ["stage_name", "result", "seed"]]
        df = df.groupby(group_by).agg({"result": "mean"}).reset_index()
        mean_tune_results = df.to_dict(orient="records")
        # maybe add std/median: df.groupby(group_by)["result"].agg(["mean", "std", "median"]).reset_index()

        # write out result
        result = dict(
            eval_results=eval_results,
            tune_results=tune_results,
            mean_eval_result=mean_eval_result,
            mean_tune_results=mean_tune_results,
        )
        pipeline_config = dict(result=result, **pipeline_config)
        with open(self.pipeline_uri, "w") as f:
            yaml.safe_dump(pipeline_config, f, sort_keys=False)

        # move tune/eval runs to summarized_area
        summarized_pipeline_area = self.work_area.summarized_area / fname_to_summarized_name(self.pipeline_uri.name)
        summarized_pipeline_area.mkdir(exist_ok=False)
        for tune_fname in tune_fnames:
            shutil.move(self.work_area.finished_area / tune_fname, summarized_pipeline_area)
        for eval_fname in eval_fnames:
            shutil.move(self.work_area.waiting_area / eval_fname, summarized_pipeline_area)
        shutil.move(self.pipeline_uri, summarized_pipeline_area)
