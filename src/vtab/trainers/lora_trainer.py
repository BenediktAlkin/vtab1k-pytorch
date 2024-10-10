from vtab.adapters.dora_linear import DoraLinear
from vtab.adapters.lora_linear import LoraLinear
from .finetune_trainer import FinetuneTrainer


class LoraTrainer(FinetuneTrainer):
    def __init__(self, lora_layer="lora", train_biases=True, lora_init="torch", **kwargs):
        super().__init__(**kwargs)
        self.lora_layer = lora_layer
        self.train_biases = train_biases
        self.lora_init = lora_init

    def setup_model(self, train_dataset, hyperparams, batch_size):
        model = super().setup_model(
            train_dataset=train_dataset,
            hyperparams=hyperparams,
            batch_size=batch_size,
        )
        rank = hyperparams.pop("rank")
        if self.lora_layer == "lora":
            layer = LoraLinear
        elif self.lora_layer == "dora":
            layer = DoraLinear
        else:
            raise NotImplementedError
        model = layer.convert(
            module=model,
            lora_rank=rank,
            lora_init=self.lora_init,
            names_to_exclude=["head"],
        )
        # freeze everything but lora matrices
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
            else:
                if self.train_biases and "bias" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        # unfreeze head
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True
        return model
