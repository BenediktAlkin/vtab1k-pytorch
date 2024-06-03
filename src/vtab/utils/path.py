from pathlib import Path


def to_path(path):
    if path is None:
        return None
    return Path(path).expanduser()
