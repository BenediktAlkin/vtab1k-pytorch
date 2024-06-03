import logging
import sys
from pathlib import Path

import torch.distributed as dist


class Logger:
    def __init__(self, log_file=None):
        # disable on non-rank0
        if dist.is_initialized() and dist.get_rank() != 0:
            log_file = None
        # init logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers = []
        formatter = logging.Formatter(
            fmt=f"%(asctime)s %(levelname).1s %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
        # init stdout handler
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.handlers.append(stdout_handler)
        # init log_file handler
        if log_file is not None:
            log_file = Path(log_file).expanduser()
            assert log_file.exists()
            file_handler = logging.FileHandler(log_file.as_posix(), mode="a")
            file_handler.setFormatter(formatter)
            logger.handlers.append(file_handler)
        self.logger = logger

    def info(self, msg):
        self.logger.info(msg)
