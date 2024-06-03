from pathlib import Path

from torch.utils.data import Dataset

from vtab.transforms import create_transform


class VtabDataset(Dataset):
    def __init__(
            self,
            split,
            transform=None,
            global_root=None,
            local_root=None,
            download=True,
            log=print,
    ):
        super().__init__()
        self.split = split
        self.transform = create_transform(transform)
        self.log = log
        global_root = self._to_path(global_root)
        local_root = self._to_path(local_root)

        # download/copy dataset
        if local_root is None:
            # no local_root -> load from global_root
            assert global_root is not None, f"either global_root or local_root is required"
            if not self.exists(global_root):
                if download:
                    self.log(f"global_root doesn't exist -> download to '{global_root.as_posix()}'")
                    self.download(global_root)
                else:
                    raise RuntimeError(
                        f"global_root doesn't exist and download=False"
                        f" -> pass download=True or download dataset manually"
                    )
            self.source_root = global_root
        else:
            # local_root -> either copy from global_root or download into local_root
            if global_root is None:
                # download directly into local_root
                if not self.exists(local_root):
                    if download:
                        self.log(f"local_root doesn't exist -> download to '{local_root.as_posix()}'")
                        self.download(local_root)
                    else:
                        raise RuntimeError(
                            f"local_root doesn't exist, global_root is None and download=False"
                            f" -> pass download=True, download dataset manually or specify a global_root"
                        )
            else:
                # copy from global_root
                if not self.exists(local_root):
                    self.log(f"copy global_root '{global_root.as_posix()}' to local_root '{local_root.as_posix()}'")
                    self.copy(src=global_root, dst=local_root)
            self.source_root = local_root
        self.log(f"initialized dataset with source_root '{self.source_root.as_posix()}' (split={self.split})")

    @staticmethod
    def _to_path(root):
        if root is not None:
            if not isinstance(root, Path):
                root = Path(root)
            root = root.expanduser()
        return root

    @property
    def num_outputs(self):
        raise NotImplementedError

    def exists(self, root):
        raise NotImplementedError

    def download(self, root):
        raise NotImplementedError

    def copy(self, src, dst):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
