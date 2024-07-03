import torch
from torch import nn


class SplitQKV(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
    ):
        super().__init__()
        self.q = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=device)
        self.k = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=device)
        self.v = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=device)

    @classmethod
    def convert(cls, module, qkv_rel_name="qkv", rel_name=None):
        module_output = module
        if type(module) == nn.Linear:
            if rel_name == qkv_rel_name:
                assert module.out_features % 3 == 0, "qkv requires output dimension that is divisible by 3"
                module_output = SplitQKV(
                    in_features=module.in_features,
                    out_features=module.out_features // 3,
                    bias=module.bias is not None,
                    device=module.weight.device,
                )
                out_dim = module.out_features // 3
                qend = kstart = out_dim
                kend = vstart = out_dim * 2
                with torch.no_grad():
                    module_output.q.weight.copy_(module.weight[:qend])
                    module_output.q.bias.copy_(module.bias[:qend])
                    module_output.k.weight.copy_(module.weight[kstart:kend])
                    module_output.k.bias.copy_(module.bias[kstart:kend])
                    module_output.v.weight.copy_(module.weight[vstart:])
                    module_output.v.bias.copy_(module.bias[vstart:])
        for name, child in module.named_children():
            module_output.add_module(
                name,
                cls.convert(
                    module=child,
                    qkv_rel_name=qkv_rel_name,
                    rel_name=name,
                ),
            )
        del module
        return module_output

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return torch.concat([q, k, v], dim=-1)
