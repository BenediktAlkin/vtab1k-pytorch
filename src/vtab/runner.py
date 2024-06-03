import os
import random
import shutil
import subprocess
from time import sleep

import yaml

from vtab.processors import create_processor
from vtab.utils.logger import Logger
from vtab.utils.naming import change_fname
from vtab.utils.work_area import WorkArea


class Runner:
    """ run all yamls from the work directory """

    def __init__(self, work_area, logger=None):
        self.logger = logger or Logger()
        self.work_area = WorkArea(work_area)

    @staticmethod
    def _fetch_yamls(uri):
        content = [uri / name for name in os.listdir(uri)]
        yamls = [entry for entry in content if entry.is_file() and entry.name.endswith(".yaml")]
        return yamls

    def run(self, devices, accelerator, num_workers, idling_tolerance=10):
        counter = 0
        idling_counter = 0

        while True:
            # fetch list of yamls to run
            work_yamls = self._fetch_yamls(self.work_area.root)
            if len(work_yamls) == 0:
                waiting_yamls = self._fetch_yamls(self.work_area.waiting_area)
                if len(waiting_yamls) == 0:
                    self.logger.info(
                        f"no yamls in '{self.work_area.root.as_posix()}' or '{self.work_area.waiting_area}'"
                        f" -> terminate"
                    )
                    break
                else:
                    # e.g. 5 tune runs but only 4 workers -> 3 workers have to wait for the last tune run to
                    # finish before the eval runs are queued
                    self.logger.info(
                        f"no yamls in '{self.work_area.root.as_posix()}' "
                        f"but {len(waiting_yamls)} in '{self.work_area.waiting_area}' -> wait for 30s"
                    )
                    sleep(30)
                    idling_counter += 1
                    # terminate after 10 minutes
                    if idling_counter > idling_tolerance:
                        break
                    continue
            idling_counter = 0

            # race condition prevention: sleep for a random interval to avoid race conditions
            sleep(random.random())

            # pick random yaml and move it to running folder
            work_yaml = random.choice(work_yamls)

            # race condition prevention: repeat if yaml was already taken in the meanwhile
            if not work_yaml.exists():
                continue

            # mark as running
            running_yaml = self.work_area.running_area / work_yaml.name
            shutil.move(work_yaml, running_yaml)
            self.logger.info(f"moved {work_yaml} to {running_yaml}")
            counter += 1

            # extract on-finish from yaml (also implicitly checks if yaml is valid)
            # noinspection PyBroadException
            try:
                with open(running_yaml) as f:
                    pipeline_config = yaml.safe_load(f)
            except Error as e:
                self.logger.info(f"couldnt load yaml {work_yaml} ({e})")
                continue
            if pipeline_config is None:
                self.logger.info(f"couldnt load yaml {work_yaml} (None)")
                continue

            # fetch current_stage
            current_stage_name = pipeline_config["current_stage"]
            stage_config = pipeline_config[current_stage_name]
            # if stage contains processor -> execute processor
            if "processor" in stage_config:
                # run processor
                processor = create_processor(pipeline_uri=running_yaml, work_area=self.work_area.root)
                processor.run()
                # move to finished folder (check because summarizer can move yaml directly to summarized_area)
                if running_yaml.exists():
                    shutil.move(running_yaml, self.work_area.finished_area / work_yaml.name)
                continue
            else:
                # run config
                if "result" in pipeline_config:
                    self.logger.info(f"{running_yaml.name} already has result -> skip")
                else:
                    # start (use seperate process to avoid potential memory leaks accumulating)
                    popen_arg_list = [
                        "python", "main_trainer.py",
                        "--pipeline_config", running_yaml.as_posix(),
                        "--devices", devices,
                        "--accelerator", accelerator,
                        "--num_workers", str(num_workers),
                    ]
                    process = subprocess.Popen(popen_arg_list)
                    self.logger.info(f"started {running_yaml.name}")
                    process.wait()

                    # reload it to check if it finished successfully (if it crashed -> no result)
                    with open(running_yaml) as f:
                        pipeline_config = yaml.safe_load(f)
                    if "result" not in pipeline_config:
                        self.logger.info(f"no result found in '{running_yaml.name}' -> move to crashed")
                        crashed_yaml = self.work_area.crashed_area / work_yaml.name
                        shutil.move(running_yaml, crashed_yaml)
                        continue

                if "next_stage" in stage_config:
                    # move to waiting folder
                    waiting_yaml = self.work_area.waiting_area / work_yaml.name
                    shutil.move(running_yaml, waiting_yaml)
                    # submit next stage
                    next_stage_name = stage_config["next_stage"]
                    assert next_stage_name in pipeline_config
                    pipeline_config.pop(current_stage_name)
                    pipeline_config["current_stage"] = next_stage_name
                    pipeline_config.pop("result")
                    submit_name = change_fname(fname=work_yaml.name, stage_name=next_stage_name)
                    with open(self.work_area.root / submit_name, "w") as f:
                        yaml.safe_dump(pipeline_config, f)
                else:
                    # move to finished folder
                    finished_yaml = self.work_area.finished_area / work_yaml.name
                    shutil.move(running_yaml, finished_yaml)

        self.logger.info(f"finished {counter} yamls from '{self.work_area.root.as_posix()}' (devices={devices})")
