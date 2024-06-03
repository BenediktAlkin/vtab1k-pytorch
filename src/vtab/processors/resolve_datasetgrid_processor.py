import yaml

from vtab.utils.select import select_set
from .base_processor import BaseProcessor


class ResolveDatasetgridProcessor(BaseProcessor):
    """ generates all pipelines for the given datasets """

    def __init__(self, paths, versions, **kwargs):
        super().__init__(**kwargs)
        self.paths = paths
        self.versions = versions

    def run(self):
        self.logger.info(f"generating '{self.pipeline_uri.as_posix()}' runs '{self.work_area.root.as_posix()}'")
        with open(self.pipeline_uri) as f:
            pipeline_config = yaml.safe_load(f)

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

        for i, version in enumerate(self.versions):
            # store to work_area
            prefix = f"{self.pipeline_uri.name.replace('.yaml', '')}_{version}"
            version_pipeline_uri = self.work_area.root / f"{prefix}.yaml"
            self.logger.info(f"creating config {i + 1}/{len(self.versions)} ({version_pipeline_uri.as_posix()})")

            # assign the version to the specified paths
            for path in self.paths:
                select_set(obj=pipeline_config, path=path, value=version)
            with open(version_pipeline_uri, "w") as f:
                yaml.safe_dump(pipeline_config, f, sort_keys=False)
