from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vtab.adapters.eva_linear import EvaLinear, BufferedPCA
from vtab.datasets import create_dataset
from vtab.models import create_model
from vtab.utils.seed import set_seed
from .base_trainer import BaseTrainer


class EvaPcaTrainer(BaseTrainer):
    def __init__(self, pooling=None, exclude=None, **kwargs):
        super().__init__(**kwargs)
        self.pooling = pooling
        self.exclude = exclude

    def train(self, hyperparams):
        # make sure environment is properly setup before training
        output_dir = Path(self.static_config["output_uri"]).expanduser()
        assert output_dir.exists() and output_dir.is_dir()



        # consume hyperparams
        model = hyperparams.pop("model")
        dataset = hyperparams.pop("dataset")
        seed = hyperparams.pop("seed")
        rank = hyperparams.pop("rank")
        batch_size = hyperparams.pop("batch_size")
        pca_batch_size = hyperparams.pop("pca_batch_size")
        pca_batch_size_head = hyperparams.pop("pca_batch_size_head", None)
        assert len(hyperparams) == 0


        # setup dataset
        dataset = create_dataset(dataset=dataset, static_config=self.static_config)

        # setup model
        model = create_model(
            model=model,
            input_shape=dataset[0][0].shape,
            num_outputs=dataset.num_outputs,
        )

        # set seed
        set_seed(seed)

        # make sure model and dataset overwrite __str__
        for name, variable in [("model", model), ("dataset", dataset)]:
            variable_str = str(variable)
            for forbidden_character in ["\n"]:
                assert forbidden_character not in variable_str, \
                    f"forbidden character in {name}, override __str__ ('{forbidden_character}' in {variable_str})"
        output_fname = (
            f"model={model}"
            f"__dataset={dataset}"
            f"__rank={rank}"
            f"_ipcabs={pca_batch_size}"
            f"_pooling={self.pooling}"
            f".th"
        )

        # fit PCA
        model = EvaLinear.convert(
            module=model,
            num_components=rank,
            pca_batch_size=pca_batch_size,
            pooling=self.pooling,
            exclude=self.exclude,
        ).to(self.device).eval()
        # handle edge case for head
        assert type(model.head) == EvaLinear
        model.head.pooling = None
        if pca_batch_size_head is not None:
            # set batch_size
            model.head.pca.batch_size = pca_batch_size_head
        # head needs n_components == num_outputs
        model.head.pca.n_components = dataset.num_outputs
        print(model)

        # propagate 1 epoch through (IncrementialPCA is fitted via EvaLinear)
        # shuffle to avoid same-class patterns in IncrementalPCA
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        for x, _ in tqdm(dataloader):
            x = x.to(self.device, non_blocking=True)
            with torch.no_grad():
                _ = model(x)

        # only need to save pca stuff
        state_dict = {key: value for key, value in model.state_dict().items() if "pca." in key}
        torch.save(state_dict, output_dir / output_fname)

        return True
