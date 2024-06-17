import torch
from torch import nn


class SplitSwigluffn(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
    ):
        super().__init__()
        self.w1 = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features=in_features, out_features=out_features, bias=bias, device=device)

    @classmethod
    def convert(cls, module, w12_rel_name="w12", rel_name=None):
        module_output = module
        if type(module) == nn.Linear:
            if rel_name == w12_rel_name:
                assert module.out_features % 2 == 0, "w12 requires output dimension that is divisible by 2"
                module_output = SplitSwigluffn(
                    in_features=module.in_features,
                    out_features=module.out_features // 2,
                    bias=module.bias is not None,
                    device=module.weight.device,
                )
                out_dim = module.out_features // 2
                with torch.no_grad():
                    module_output.w1.weight.copy_(module.weight[:out_dim])
                    module_output.w1.bias.copy_(module.bias[:out_dim])
                    module_output.w2.weight.copy_(module.weight[out_dim:])
                    module_output.w2.bias.copy_(module.bias[out_dim:])
        for name, child in module.named_children():
            module_output.add_module(
                name,
                cls.convert(
                    module=child,
                    w12_rel_name=w12_rel_name,
                    rel_name=name,
                ),
            )
        del module
        return module_output

    def forward(self, x):
        w1 = self.w1(x)
        w2 = self.w2(x)
        return torch.concat([w1, w2], dim=-1)
