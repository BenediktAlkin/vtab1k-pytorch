# VTAB-1K pytorch

Plug &amp; Play implementation of the Visual Task Adaptation 1K Benchmark (VTAB-1K) in pytorch

# Setup

Instructions to setup environment/code and data are provided in [SETUP_DATA.md](https://github.com/BenediktAlkin/vtab1k-pytorch/blob/main/SETUP_DATA.md) and [SETUP_ENV.md](https://github.com/BenediktAlkin/vtab1k-pytorch/blob/main/SETUP_ENV.md).

# Evaluation Protocol

The default evaluation protocol first tunes hyperparameters on the validation set and then trains a model on the union
of train and validation set. This model is then evaluated on the testset. Each run can use multiple seeds.
As this results in a lot of runs (e.g. by using 5 seeds and 5 learning rates, this results in 30 training runs),
a evaluation is defined via a pipeline. The pipeline will use a work directory where it will keep track of what
runs are queued/running/finished. 

To start an evaluation, copy a pipeline to a work directory (by default `./work`) and run 
`python main_runner.py`. By default it will use `./work` as work_area (change via `--work_area <WORK_AREA>`) and
the GPU with index 0 (change via `--devices <GPU_INDEX>` e.g. `--devices 3`).

You can easily parallelize the whole pipeline by simply starting multiple runners on different GPUs with the same 
`work_area`.

A pipeline with a small model to test if your setup is correctly setup can be found in `pipelines/debug/debug.yaml`.

# Overview of training pipeline

The `main_runner.py` script will continuously fetch a new yaml file from the `work_area` and move it into the subfolder
`work_area/running`. It will then process the yaml (which is either training a model, checking if other runs finished
to create a summary or generating new yamls). After the yaml is finished, it will write a result into the file and move 
it to `work_area/finished`.
In the case where a run needs other runs in order to be considered `finished` (e.g. if hyperparameters are tuned on the
validation set, all hyperparameter search runs need to be finished before the evaluation runs on the testset are 
started), it will instead be moved into `work_area/waiting` and wait for the other runs to finish. If a run crashes, 
it will be moved into `work_area/crashed`. After all runs are finished, a summary is created in `work_area/summarized`
which can then be easily parsed with `main_parse_results.py`. 

If for whatever reasons runs get stuck in `work_area/running` or crash and are moved into `work_area/crashed`, you can
simply restart them by moving them to `work_area`.


# Results

The results of each run are written into the corresponding yaml.
The script `main_parse_results.py` summarizes the result of a full VTAB-1K pipeline into a single value for each dataset.

# SLURM
- Copy `sbatch_config_example.yaml` as `sbatch_config.yaml` and adjust the values to your setup.
- Copy `template_sbatch_example.sh` as `template_sbatch.sh` and adjust the values to your setup.
- Submit a job via `python main_sbatch_runner.py --time 5:00:00`

The `main_sbatch_runner.py` will do the following steps:
- fetch the `sbatch_config.yaml` 
- include the content of `sbatch_config.yaml` into `template_sbatch.sh`
- submit the patched template via `sbatch`

If you want to parallelize the evaluations, we recommend to submit many jobs with only 1 GPU. 
While `main_sbatch_runner.py` supports submitting jobs with many GPUs (e.g. `--gpus 16` for 16 GPUs), runners will
terminate if they are idle for too long which will lead to GPUs idling. If you submit only single GPU jobs, this will
not happen as the job is simply terminated if it has to wait for other runs to finish.


If you heavily parallelize the evaluations, chances are that some runs will crash (hardware failures, ...) and 
will therefore either be moved into the `crashed` folder or be stuck in the `running` folder.
In this case, simply move the crashed yamls back into the `work` folder and start the pipeline again.