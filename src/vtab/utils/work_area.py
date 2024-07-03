from pathlib import Path


class WorkArea:
    def __init__(self, root):
        self.root = Path(root).expanduser()
        if self.root.exists():
            assert self.root.is_dir(), f"'{self.root.as_posix()}' is not a directory"
        else:
            self.root.mkdir(parents=True, exist_ok=True)
        self.running_area.mkdir(exist_ok=True)
        self.finished_area.mkdir(exist_ok=True)
        self.waiting_area.mkdir(exist_ok=True)
        self.crashed_area.mkdir(exist_ok=True)
        self.summarized_area.mkdir(exist_ok=True)

    @property
    def running_area(self):
        return self.root / "running"

    @property
    def finished_area(self):
        return self.root / "finished"

    @property
    def waiting_area(self):
        return self.root / "waiting"

    @property
    def crashed_area(self):
        return self.root / "crashed"

    @property
    def summarized_area(self):
        return self.root / "summarized"
