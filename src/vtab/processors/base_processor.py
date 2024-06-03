from pathlib import Path

from vtab.utils.work_area import WorkArea
from vtab.utils.logger import Logger

class BaseProcessor:
    """ selects best hyperparameters from tuning runs and generates eval yamls """

    def __init__(self, work_area, pipeline_uri, logger=None):
        self.logger = logger or Logger()
        self.work_area = WorkArea(work_area)
        self.pipeline_uri = Path(pipeline_uri).expanduser().with_suffix(".yaml")
        assert self.pipeline_uri.exists(), f"'{self.pipeline_uri.as_posix()}' doesn't exist"

    def run(self):
        raise NotImplementedError
