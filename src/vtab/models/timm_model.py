import timm
from torch import nn


class TimmModel(nn.Module):
    def __init__(self, identifier, input_shape, num_outputs):
        # "deit3_small_patch16_224"
        super().__init__()
        assert input_shape == (3, 224, 224)
        self.model = timm.create_model(identifier, pretrained=True)
        self.model.head = nn.Linear(self.dim, num_outputs)
        nn.init.trunc_normal_(self.model.head.weight, std=2e-5)
        nn.init.zeros_(self.model.head.bias)

    def forward(self, x):
        return self.model(x)

    @property
    def dim(self):
        if hasattr(self.model, "embed_dim"):
            return self.model.embed_dim
        if hasattr(self.model, "num_features"):
            return self.model.num_features
        raise NotImplementedError
