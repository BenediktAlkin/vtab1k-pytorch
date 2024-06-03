import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics.functional.classification import multiclass_accuracy
from tqdm import tqdm

from vtab.datasets import create_dataset
from vtab.models import create_model
from vtab.optims import create_optim
from vtab.schedules import create_schedule
from vtab.utils.autocast import get_grad_scaler_and_autocast_context
from vtab.utils.formatting import to_si
from vtab.utils.infinite_batch_sampler import InfiniteBatchSampler
from vtab.utils.seed import set_seed
from .base_trainer import BaseTrainer


class FinetuneTrainer(BaseTrainer):
    def setup_model(self, train_dataset, hyperparams, batch_size):
        model = create_model(
            model=hyperparams.pop("model"),
            input_shape=train_dataset[0][0].shape,
            num_outputs=train_dataset.num_outputs,
        )
        return model

    def train(self, hyperparams):
        for key, value in hyperparams.items():
            self.logger.info(f"{key}: {value}")

        # set seed
        set_seed(hyperparams.pop("seed"))

        # create dataset
        train_dataset = create_dataset(dataset=hyperparams.pop("train_dataset"), static_config=self.static_config)
        eval_dataset = create_dataset(dataset=hyperparams.pop("eval_dataset"), static_config=self.static_config)

        # create model
        batch_size = hyperparams.pop("batch_size")
        model = self.setup_model(
            train_dataset=train_dataset,
            hyperparams=hyperparams,
            batch_size=batch_size,
        ).to(self.device)

        # create schedule
        epochs = hyperparams.pop("epochs")
        updates_per_epoch = len(train_dataset) // batch_size
        total_updates = updates_per_epoch * epochs
        schedule = create_schedule(kind=hyperparams.pop("schedule"), total_updates=total_updates)

        # init autocast and grad scaler
        grad_scaler, autocast_context = get_grad_scaler_and_autocast_context(
            precision=hyperparams.pop("precision"),
            device=self.device,
        )

        # create optim
        lr = hyperparams.pop("lr")
        optim = create_optim(
            optim=hyperparams.pop("optim"),
            model=model,
            lr=lr,
            batch_size=batch_size,
            schedule=schedule,
        )

        # check unconsumed hyperparameters
        accumulation_steps = hyperparams.pop("accumulation_steps", 1)
        assert len(hyperparams) == 0, f"unused hyperparameters {list(hyperparams.keys())}"

        #
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        num_params = num_trainable_params + num_frozen_params
        self.logger.info(f"#trainable: {to_si(num_trainable_params)}")
        self.logger.info(f"#frozen: {to_si(num_frozen_params)}")
        self.logger.info(f"%trainable: {num_trainable_params / num_params * 100:.2f}%")
        self.logger.info(f"#total: {to_si(num_trainable_params + num_frozen_params)}")

        # init dataloader
        train_dataloader = DataLoader(
            dataset=train_dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=InfiniteBatchSampler(
                epochs=epochs,
                sampler=RandomSampler(train_dataset),
                batch_size=batch_size,
                drop_last=True,
            ),
        )
        train_iterator = iter(train_dataloader)

        # train
        model.train()
        for update in tqdm(range(epochs * updates_per_epoch)):
            optim.zero_grad()
            x, y = next(train_iterator)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            assert len(x) % accumulation_steps == 0
            samples_per_accumulation_step = len(x) // accumulation_steps
            for i in range(accumulation_steps):
                start = i * samples_per_accumulation_step
                end = start + samples_per_accumulation_step
                with autocast_context:
                    y_hat = model(x[start:end])
                    loss = F.cross_entropy(y_hat, y[start:end])
                grad_scaler.scale(loss / accumulation_steps).backward()
            optim.step(update=update)

        # eval
        eval_dataloader = DataLoader(
            dataset=eval_dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        y_hats = []
        ys = []
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(eval_dataloader):
                x = x.to(self.device, non_blocking=True)
                y_hat = model(x)
                y_hats.append(y_hat.cpu())
                ys.append(y.clone())
        y_hat = torch.concat(y_hats)
        y = torch.concat(ys)
        _, num_classes = y_hat.shape
        acc = multiclass_accuracy(
            preds=y_hat,
            target=y,
            top_k=1,
            num_classes=num_classes,
            average="micro",
        ).item()
        return acc
