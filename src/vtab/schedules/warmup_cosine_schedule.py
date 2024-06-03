import math

import torch


class WarmupCosineSchedule:
    def __init__(self, warmup_updates, total_updates):
        assert warmup_updates <= total_updates
        self.warmup_updates = warmup_updates
        self.total_updates = total_updates
        warmup_progress = torch.linspace(0, 1, warmup_updates + 1)[1:].tolist()
        cosine_updates = total_updates - warmup_updates
        cosine_progress = [
            ((1 + math.cos(math.pi * (i / (cosine_updates - 1)))) / 2)
            for i in range(1, cosine_updates + 1)
        ]
        self.progress = warmup_progress + cosine_progress

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
