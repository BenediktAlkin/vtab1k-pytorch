import os
import shutil

import torch
from torch.utils.data import Subset
from torchvision.datasets.cifar import CIFAR100

from .vtab_dataset import VtabDataset


class Cifar100(VtabDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset = CIFAR100(
            root=self.source_root,
            download=False,
            train=self.split != "test",
            transform=self.transform,
        )
        if self.split in ["train", "test"]:
            pass
        elif self.split == "train800":
            indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(0))[:800].tolist()
            self.dataset = Subset(self.dataset, indices)
            raise NotImplementedError
        elif self.split == "val200":
            indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(0))[800:1000].tolist()
            self.dataset = Subset(self.dataset, indices)
            raise NotImplementedError
        elif self.split == "train800val200":
            indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(0))[:1000].tolist()
            self.dataset = Subset(self.dataset, indices)
            raise NotImplementedError
        else:
            raise NotImplementedError

    @property
    def num_outputs(self):
        return 100

    def exists(self, root):
        return (root / "cifar-100-python").exists()

    def download(self, root):
        _ = CIFAR100(root=root, download=True)
        os.remove(root / "cifar-100-python.tar.gz")

    def copy(self, src, dst):
        shutil.copytree(src / "cifar-100-python", dst / "cifar-100-python")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
