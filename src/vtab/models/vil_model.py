from torch import nn


class VilModel(nn.Module):
    def __init__(self, identifier, input_shape, num_outputs):
        # "vil2-tiny"
        # "vil2-tiny-e400"
        # "vil2-small"
        # "vil2-base"
        super().__init__()
        assert input_shape == (3, 224, 224)
        self.model = torch.hub.load("nx-ai/vision-lstm", identifier)
        self.model.head = nn.Linear(self.dim, num_outputs)
        nn.init.trunc_normal_(self.model.head.weight, std=2e-5)
        nn.init.zeros_(self.model.head.bias)

    def forward(self, x):
        return self.model(x)

    @property
    def dim(self):
        return self.model.dim
