import math

import torch


class WarmupLinearSchedule:
    def __init__(self, warmup_updates, total_updates):
        assert warmup_updates <= total_updates
        self.warmup_updates = warmup_updates
        self.total_updates = total_updates
        warmup_progress = torch.linspace(0, 1, warmup_updates + 1)[1:].tolist()
        linear_updates = total_updates - warmup_updates
        linear_progress = torch.linspace(1, 0, linear_updates + 1)[:-1].tolist()
        self.progress = warmup_progress + linear_progress

    def __len__(self):
        return len(self.progress)

    def __getitem__(self, idx):
        return self.progress[idx]

    def __str__(self):
        return (
            f"{type(self).__name__}("
            f"warmup_updates={self.warmup_updates},"
            f"total_updates={self.total_updates},"
            f"warmup_percent={self.warmup_updates / self.total_updates:.2f})"
        )
