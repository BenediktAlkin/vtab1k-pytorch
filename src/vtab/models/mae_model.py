import torch
from torch import nn

from vtab.adapters.split_qkv import SplitQKV


class MaeModel(nn.Module):
    def __init__(self, identifier, input_shape, num_outputs, split_qkv=False, freeze_num_blocks=None):
        # "in1k_mae_b16"
        # "in1k_mae_l16"
        # "in1k_mae_h14"
        # "in1k_mae_twob14"
        super().__init__()
        assert input_shape == (3, 224, 224)
        model = torch.hub.load("BenediktAlkin/torchhub-ssl", identifier)
        if split_qkv:
            model = SplitQKV.convert(model)
        self.model = model
        self.head = nn.Linear(self.model.dim, num_outputs)
        nn.init.trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)

        if freeze_num_blocks is not None:
            to_freeze = [self.model.patch_embed, self.model.pos_embed, self.model.cls_tokens]
            if identifier.startswith("d2v2"):
                to_freeze.append(self.model.norm)
            for i in range(freeze_num_blocks):
                to_freeze.append(self.model.blocks[i])
            for module in to_freeze:
                for name, p in module.named_parameters():
                    p.requires_grad = False

    def forward(self, x):
        return self.head(self.model(x)[:, 0])

    @property
    def dim(self):
        return self.model.dim
