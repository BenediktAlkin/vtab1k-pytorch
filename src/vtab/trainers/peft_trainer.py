import re

from torch import nn

from vtab.models.dinov2_peft_model import Dinov2PeftModel
from .finetune_trainer import FinetuneTrainer


class PeftTrainer(FinetuneTrainer):
    class HuggingfaceWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x).logits

    def setup_model(self, train_dataset, hyperparams, batch_size):
        # setup peft config
        peft_config = hyperparams.pop("peft_config")
        kind = peft_config["kind"]
        if kind == "lora_config":
            rank = hyperparams.pop("rank")
            peft_config["r"] = rank
        elif kind == "boft_config":
            # parse "m2b8" to dict(m=2, b=8)
            boftparams = hyperparams.pop("boftparams")
            matches = re.compile(r"([a-zA-Z])(\d+)").findall(boftparams)
            boftparams = {letter: int(number) for letter, number in matches}
            peft_config["boft_block_size"] = boftparams["b"]
            peft_config["boft_n_butterfly_factor"] = boftparams["m"]
        elif kind == "adalora_config":
            epochs = hyperparams["epochs"]
            updates_per_epoch = len(train_dataset) // batch_size
            total_updates = updates_per_epoch * epochs
            peft_config["tinit"] = int(total_updates * 0.1)
            peft_config["tfinal"] = int(total_updates * 0.2)
            rank = hyperparams.pop("rank")
            peft_config["init_r"] = rank + 4
            peft_config["target_r"] = rank
            peft_config["total_step"] = total_updates
        else:
            raise NotImplementedError

        model = hyperparams.pop("model")
        model_kind = model.pop("kind")
        if model_kind == "dinov2_peft_model":
            model = Dinov2PeftModel(
                input_shape=train_dataset[0][0].shape,
                num_outputs=train_dataset.num_outputs,
                peft_config=peft_config,
                **model,
            )
        else:
            raise NotImplementedError

        return model
