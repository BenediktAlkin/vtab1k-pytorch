import timm
from torch import nn


class TimmModel(nn.Module):
    def __init__(self, identifier, input_shape, num_outputs):
        # "deit3_small_patch16_224"
        super().__init__()
        assert input_shape == (3, 224, 224)
        self.model = timm.create_model(identifier, pretrained=True)
        self.model.head = nn.Linear(self.model.embed_dim, num_outputs)
        nn.init.trunc_normal_(self.model.head.weight, std=2e-5)
        nn.init.zeros_(self.model.head.bias)

    def forward(self, x):
        return self.model(x)

    @property
    def dim(self):
        return self.model.embed_dim
