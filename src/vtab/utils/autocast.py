import torch
from torch.cuda.amp import GradScaler


class NoopContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class NoopGradScaler:
    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        pass

    @staticmethod
    def step(optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    @staticmethod
    def update():
        pass


def get_grad_scaler_and_autocast_context(precision, device):
    if precision in ["fp32", "float32", torch.float32]:
        return NoopGradScaler(), NoopContext()
    if precision in ["bf16", "bfloat16", torch.bfloat16]:
        return NoopGradScaler(), torch.autocast(str(device).split(":")[0], dtype=torch.bfloat16)
    if precision in ["fp16", "float16", torch.float16]:
        return GradScaler(), torch.autocast(str(device).split(":")[0], dtype=torch.float16)
    raise NotImplementedError
