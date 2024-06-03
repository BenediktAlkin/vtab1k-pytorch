from torch import nn


class LinearModel(nn.Module):
    def __init__(self, input_shape, num_outputs, latent_dim=8):
        super().__init__()
        assert len(input_shape) == 3 and input_shape[0] == 3 and input_shape[1] >= 32 and input_shape[1] >= 32
        self.latent_dim = latent_dim
        self.stem = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=8), nn.Flatten())
        self.encoder = nn.Linear(3 * 8 * 8, latent_dim)
        self.head = nn.Linear(latent_dim, num_outputs)

    def __str__(self):
        return "linear"

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.head(x)
        return x

    @property
    def dim(self):
        return self.latent_dim
