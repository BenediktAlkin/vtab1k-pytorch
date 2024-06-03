import os
import shutil
import zipfile
from pathlib import Path

from torchvision.datasets.folder import default_loader

from .vtab_dataset import VtabDataset


class Vtab1kDataset(VtabDataset):
    def __init__(
            self,
            static_config=None,
            version=None,
            global_root=None,
            local_root=None,
            **kwargs,
    ):
        self.version = version
        # fetch global/local roots from static_config
        if global_root is None and local_root is None:
            assert static_config is not None, f"no global_root, local_root or static_config defined"
            assert version is not None, f"deriving dataset paths via static_config requires VTAB version (e.g. cifar)"
            # fetch global_root
            assert "vtab1k" in static_config
            global_root = Path(static_config["vtab1k"]).expanduser() / version
            if "local_root" in static_config:
                local_root = Path(static_config["local_root"]).expanduser() / "vtab-1k" / version
        super().__init__(global_root=global_root, local_root=local_root, **kwargs)
        with open(self.source_root / f"{self.split}.txt") as f:
            lines = f.readlines()
        self.fnames = []
        self.labels = []
        for line in lines:
            fname, label = line.strip().split(" ")
            self.fnames.append(fname)
            self.labels.append(int(label))
        self._num_outputs = max(self.labels) + 1
        if self.split == "train800":
            assert len(self) == 800
        elif self.split == "val200":
            assert len(self) == 200
        elif self.split == "train800val200":
            assert len(self) == 1000
        elif self.split == "test":
            if version is None:
                pass
            elif version == "caltech101":
                assert len(self) == 6085
            elif version == "cifar":
                assert len(self) == 10000
            elif version == "clevr_count":
                assert len(self) == 15000
            elif version == "clevr_dist":
                assert len(self) == 15000
            elif version == "diabetic_retinopathy":
                assert len(self) == 42670
            elif version == "dmlab":
                assert len(self) == 22735
            elif version == "dsprites_loc":
                assert len(self) == 73728
            elif version == "dsprites_ori":
                assert len(self) == 73728
            elif version == "dtd":
                assert len(self) == 1880
            elif version == "eurosat":
                assert len(self) == 5400
            elif version == "kitti":
                assert len(self) == 711
            elif version == "oxford_flowers102":
                assert len(self) == 6149
            elif version == "oxford_iiit_pet":
                assert len(self) == 3669
            elif version == "patch_camelyon":
                assert len(self) == 32768
            elif version == "resisc45":
                assert len(self) == 6300
            elif version == "smallnorb_azi":
                assert len(self) == 12150
            elif version == "smallnorb_ele":
                assert len(self) == 12150
            elif version == "sun397":
                assert len(self) == 21750
            elif version == "svhn":
                assert len(self) == 26032
            else:
                raise NotImplementedError(f"invalid version '{version}'")
        else:
            raise NotImplementedError(f"invalid split '{self.split}'")

    def __str__(self):
        return f"{self.version}_{self.split}"

    @property
    def num_outputs(self):
        return self._num_outputs

    def exists(self, root):
        if not (root / f"{self.split}.txt").exists():
            return False
        if not (root / "images").exists():
            return False
        if not (root / "images" / self.split).exists():
            return False
        return True

    def download(self, root):
        if root.parent.exists():
            content = os.listdir(root.parent)
            if len(content) > 0:
                if len(content) != 1 or content[0] != "vtab-1k.zip":
                    raise RuntimeError(f"'{root.parent}' exists and is not empty")
        if (root.parent / "vtab-1k.zip").exists():
            self.log(
                f"'{root.as_posix()}' doesn't exist but zip was already downloaded ('{root.parent / 'vtab-1k.zip'}') "
                f"-> skip download"
            )
        else:
            self.log(
                f"'{root.as_posix()}' doesn't exist "
                f"-> download self-contained VTAB-1K dataset to {root.parent.as_posix()}"
            )
            root.parent.mkdir(exist_ok=True, parents=True)
            try:
                import gdown
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "No module named 'gdown' -> install via 'pip install gdown' "
                    "(required for downloading self-contained VTAB-1K dataset from google drive)"
                )
            gdown.download(id="1yZKwiKdsBzTfBgnStRveYMokc7GMMd5p", output=(root.parent / "vtab-1k.zip").as_posix())
        import zipfile
        self.log(f"extracting downloaded dataset")
        with zipfile.ZipFile(root.parent / "vtab-1k.zip", "r") as f:
            f.extractall(root.parent)
        # zip contains a single folder -> would create .../vtab-1k/vtab-1k/cifar -> remove the folder
        for fname in os.listdir(root.parent / "vtab-1k"):
            shutil.move(root.parent / "vtab-1k" / fname, root.parent / fname)
        (root.parent / "vtab-1k").rmdir()
        os.remove(root.parent / "vtab-1k.zip")

    def copy(self, src, dst):
        # check if zip exists
        src_zip = src.with_suffix(".zip")
        if src_zip.exists():
            # unzip (typically faster than copying individual files)
            with zipfile.ZipFile(src_zip, "r") as f:
                f.extractall(dst.parent)
        else:
            # copy individual files
            shutil.copytree(src, dst)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        x = default_loader(self.source_root / self.fnames[idx])
        if self.transform is not None:
            x = self.transform(x)
        y = self.labels[idx]
        return x, y
