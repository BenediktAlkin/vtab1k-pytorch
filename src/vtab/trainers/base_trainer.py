import torch

from vtab.utils.logger import Logger


class BaseTrainer:
    def __init__(
            self,
            device=0,
            accelerator="gpu",
            num_workers=10,
            pin_memory=True,
            cudnn_benchmark=True,
            cudnn_deterministic=False,
            static_config=None,
            logger=None,
    ):
        super().__init__()
        self.logger = logger or Logger()
        self.logger.info(f"{type(self).__name__}")
        self.static_config = static_config

        # init device
        if accelerator == "cpu":
            self.device = torch.device("cpu")
        elif accelerator == "gpu":
            self.device = torch.device(f"cuda:{device}")
        else:
            raise NotImplementedError(f"invalid accelerator '{accelerator}' (use 'cpu' or 'gpu')")
        self.logger.info(f"device: {self.device}")

        # dataloading
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # cudnn
        self.cudnn_benchmark = cudnn_benchmark
        self.cudnn_deterministic = cudnn_deterministic
        if accelerator == "gpu":
            if cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                assert not cudnn_deterministic
                self.logger.info(f"enabled cudnn benchmark")
            else:
                torch.backends.cudnn.benchmark = False
                self.logger.info(f"disabled cudnn benchmark")
            if cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                self.logger.info(f"enabled cudnn deterministic")
            else:
                torch.backends.cudnn.deterministic = False
                self.logger.info(f"disabled cudnn deterministic")

    def train(self, hyperparams):
        raise NotImplementedError
