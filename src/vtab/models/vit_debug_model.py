from kappamodules.vit import VitPatchEmbed, VitBlock, VitClassTokens
from torch import nn
import einops

class VitDebugModel(nn.Module):
    def __init__(self, input_shape, num_outputs, latent_dim=8):
        super().__init__()
        assert len(input_shape) == 3 and input_shape[0] == 3 and input_shape[1] >= 32 and input_shape[1] >= 32
        self.latent_dim = latent_dim

        class Vit(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = VitPatchEmbed(
                    dim=latent_dim,
                    num_channels=3,
                    resolution=(input_shape[1], input_shape[1]),
                    patch_size=16,
                )
                self.cls_tokens = VitClassTokens(dim=latent_dim)
                self.blocks = nn.ModuleList([VitBlock(dim=latent_dim, num_heads=1) for _ in range(2)])

            def forward(self, x):
                x = self.patch_embed(x)
                x = einops.rearrange(x, "b ... c -> b (...) c")
                x = self.cls_tokens(x)
                for block in self.blocks:
                    x = block(x)
                return x[:, 0]

        self.encoder = Vit()
        self.head = nn.Linear(latent_dim, num_outputs)

    def __str__(self):
        return "vit_debug"

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x

    @property
    def dim(self):
        return self.latent_dim
