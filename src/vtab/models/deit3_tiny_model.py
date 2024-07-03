import einops
import torch
from kappamodules.transformer import PrenormBlock
from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d, VitClassTokens
from torch import nn


class PrenormVit(nn.Module):
    def __init__(
            self,
            patch_size,
            dim,
            depth,
            num_heads,
            input_shape=(3, 224, 224),
            mlp_hidden_dim=None,
            drop_path_rate=0.,
            drop_path_decay=True,
            num_cls_tokens=1,
            layerscale=1e-4,
            num_outputs=1000,
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.input_shape = input_shape
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.eps = eps

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            num_channels=input_shape[0],
            resolution=input_shape[1:],
            patch_size=patch_size,
        )

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # 0, 1 or more cls tokens
        self.cls_tokens = VitClassTokens(dim=dim, num_tokens=num_cls_tokens)

        # stochastic depth
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # blocks
        self.blocks = nn.ModuleList([
            PrenormBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_ctor=nn.LayerNorm,
                drop_path=dpr[i],
                layerscale=layerscale,
                eps=eps,
            )
            for i in range(depth)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(dim, eps=eps),
            nn.Linear(dim, num_outputs),
        )

        self.output_shape = (self.patch_embed.num_patches + self.cls_tokens.num_tokens, dim)

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)
        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")
        # add cls token
        x = self.cls_tokens(x)
        # apply blocks
        for blk in self.blocks:
            x = blk(x)
        # last norm
        x = self.head(x[:, 0])
        return x


class Deit3TinyModel(nn.Module):
    def __init__(self, input_shape, num_outputs):
        super().__init__()
        assert input_shape == (3, 224, 224)
        self.model = PrenormVit(
            patch_size=16,
            dim=192,
            depth=12,
            num_heads=3,
            num_outputs=1000,
        )
        sd = torch.hub.load_state_dict_from_url(
            "https://ml.jku.at/research/vision_lstm/download/vit_tiny16_e800_in1k_deit3reimpl.th",
            map_location="cpu",
        )["state_dict"]
        self.model.load_state_dict(sd)
        self.model.head[1] = nn.Linear(self.model.dim, num_outputs)
        nn.init.trunc_normal_(self.model.head[1].weight, std=2e-5)
        nn.init.zeros_(self.model.head[1].bias)

    def forward(self, x):
        return self.model(x)

    @property
    def dim(self):
        return self.model.dim
