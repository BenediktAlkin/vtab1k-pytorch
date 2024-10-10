import einops
import torch
from sklearn.decomposition import IncrementalPCA, PCA
from torch import nn


class _BufferedIncrementalPCA(IncrementalPCA):
    """
    IncrementalPCA that accumulates batches if number of
    samples for partial_fit is smaller than num_components
    """

    def __init__(self, n_components, batch_size):
        super().__init__(n_components=n_components, batch_size=batch_size)
        self.buffer = []
        self.buffer_size = 0

    def partial_fit(self, x):
        assert torch.is_tensor(x) and x.ndim == 2
        x = x.cpu()
        self.buffer.append(x)
        self.buffer_size += len(x)
        if self.buffer_size >= self.batch_size and self.buffer_size >= self.n_components:
            x = torch.concat(self.buffer).numpy()
            assert len(x) > 0
            super().partial_fit(x)
            self.buffer.clear()
            self.buffer_size = 0


class BufferedPCA(PCA):
    """
    PCA that accumulates batches and does .fit if batch_size is reached
    throws exception if more batches are attempted to be buffered/fitted after calling .fit
    """

    def __init__(self, n_components, batch_size):
        super().__init__(n_components=n_components)
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_size = 0

    def partial_fit(self, x):
        assert torch.is_tensor(x) and x.ndim == 2
        assert not hasattr(self, "components_")
        x = x.cpu()
        self.buffer.append(x)
        self.buffer_size += len(x)
        if self.buffer_size >= self.n_components:
            x = torch.concat(self.buffer).numpy()
            super().fit(x)
            self.buffer.clear()
            self.buffer_size = 0


class EvaLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_components: int,
            pca_batch_size: int,
            pooling: int = None,
            bias: bool = True,
            **kwargs,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias, **kwargs)
        self.num_components = num_components
        self.pca_batch_size = pca_batch_size
        self.pooling = pooling
        self.pca = _BufferedIncrementalPCA(n_components=num_components, batch_size=pca_batch_size)

    @classmethod
    def convert(cls, module, num_components, pca_batch_size, pooling=None, exclude=None, abs_name=None):
        exclude = exclude or []
        module_output = module
        if type(module) == nn.Linear:
            if abs_name is None or abs_name not in exclude:
                module_output = EvaLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    num_components=num_components,
                    pca_batch_size=pca_batch_size,
                    pooling=pooling,
                )
                module_output.weight = module.weight
                module_output.bias = module.bias
        for name, child in module.named_children():
            module_output.add_module(
                name,
                cls.convert(
                    module=child,
                    num_components=num_components,
                    pca_batch_size=pca_batch_size,
                    pooling=pooling,
                    exclude=exclude,
                    abs_name=name if abs_name is None else f"{abs_name}.{name}",
                ),
            )
        del module
        return module_output

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        if not hasattr(self.pca, "components_"):
            print(f"no samples were propagated to extract features for PCA (prefix={prefix})")
            print(f"buffer size: {self.pca.buffer_size}")
            return {}
        state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        state_dict[f"{prefix}pca.components"] = torch.from_numpy(self.pca.components_)
        state_dict[f"{prefix}pca.singular_values"] = torch.from_numpy(self.pca.singular_values_)
        state_dict[f"{prefix}pca.explained_variance"] = torch.from_numpy(self.pca.explained_variance_)
        state_dict[f"{prefix}pca.explained_variance_ratio"] = torch.from_numpy(self.pca.explained_variance_ratio_)
        state_dict[f"{prefix}pca.mean"] = torch.from_numpy(self.pca.mean_)
        state_dict[f"{prefix}pca.var"] = torch.from_numpy(self.pca.var_)
        state_dict[f"{prefix}pca.noise_variance"] = torch.tensor(self.pca.noise_variance_)
        state_dict[f"{prefix}pca.n_samples_seen"] = torch.tensor(self.pca.n_samples_seen_)
        state_dict[f"{prefix}pca.batch_size"] = torch.tensor(self.pca.batch_size)
        return state_dict

    def forward(self, x):
        assert not torch.is_grad_enabled()
        assert not x.requires_grad
        if self.pooling is None:
            pca_input = einops.rearrange(x, "b ... dim -> (b ...) dim")
        elif self.pooling == "class_token":
            if x.ndim == 3:
                pca_input = x[:, 0]
            else:
                assert x.ndim == 2
                pca_input = x
        else:
            raise NotImplementedError
        self.pca.partial_fit(pca_input)
        return super().forward(x)
