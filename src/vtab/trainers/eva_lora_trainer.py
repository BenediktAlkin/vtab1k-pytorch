import torch
from pathlib import Path
from vtab.adapters.lora_linear import LoraLinear
from vtab.adapters.dora_linear import DoraLinear
from .finetune_trainer import FinetuneTrainer
from scipy.stats import ortho_group

class EvaLoraTrainer(FinetuneTrainer):
    def __init__(
            self,
            pca_rel_uri,
            lora_layer="lora",
            whiten_pca=False,
            distribute_ranks=False,
            init_head_with_pc=False,
            pca_rel_uri_format_kwargs=None,
            random_rotation=False,
            random_permutation=False,
            train_biases=True,
            names_to_exclude=None,
            init_with_pcs=True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.pca_rel_uri = pca_rel_uri
        self.lora_layer = lora_layer
        self.whiten_pca = whiten_pca
        self.distribute_ranks = distribute_ranks
        self.init_head_with_pc = init_head_with_pc
        self.pca_rel_uri_format_kwargs = pca_rel_uri_format_kwargs or {}
        self.random_rotation = random_rotation
        self.random_permutation = random_permutation
        self.train_biases = train_biases
        self.init_with_pcs = init_with_pcs
        self.names_to_exclude = names_to_exclude or []

    def setup_model(self, train_dataset, hyperparams, batch_size):
        model = super().setup_model(
            train_dataset=train_dataset,
            hyperparams=hyperparams,
            batch_size=batch_size,
        )
        # convert to LoraLinear
        rank = hyperparams.pop("rank")
        if self.lora_layer == "lora":
            layer = LoraLinear
        elif self.lora_layer == "dora":
            layer = DoraLinear
        else:
            raise NotImplementedError
        model = layer.convert(
            module=model,
            lora_rank=rank,
            names_to_exclude=self.names_to_exclude + ["head"],
        )

        # init lora_A with PCA components
        output_dir = Path(self.static_config["output_uri"]).expanduser()
        pca = torch.load(output_dir / self.pca_rel_uri.format(**self.pca_rel_uri_format_kwargs))
        # TODO legacy (first implementation uses ipca instead of .pca
        pca = {key.replace("ipca", "pca"): value for key, value in pca.items()}
        # trim number of pcs from PCA
        rho_mode = hyperparams.pop("rhomode", None)
        if rho_mode is not None:
            assert "rho" not in hyperparams
            if rho_mode == "inf":
                max_num_pcs = 9999999
            elif rho_mode == "tworank":
                max_num_pcs = 2 * rank
            elif rho_mode == "fourrank":
                max_num_pcs = 4 * rank
            else:
                raise NotImplementedError
        else:
            max_num_pcs = hyperparams.pop("rho", None)
        if max_num_pcs is not None:
            pca = {
                key: value[:max_num_pcs]
                for key, value in pca.items()
                if (
                        key.endswith(".components")
                        or key.endswith(".singular_values")
                        or key.endswith(".explained_variance")
                        or key.endswith(".explained_variance_ratio")
                )
            }
        components_as_sd = {
            key.replace(".pca.components", ".lora_A"): value
            for key, value in pca.items()
            if key.endswith(".pca.components") and not key.startswith("head.")
        }
        if self.init_head_with_pc:
            components_as_sd["head.weight"] = pca["head.pca.components"]
        if self.distribute_ranks:
            # get total number of ranks
            budget = rank * len([key for key in components_as_sd.keys() if key.endswith("lora_A")])
            # find the threshold for the explained variance
            all_explained_variance_ratios = torch.concat(
                [
                    pca[key]
                    for key in pca.keys()
                    if key.endswith(".pca.explained_variance_ratio") and not key.startswith("head.")
                ]
            )
            threshold = all_explained_variance_ratios.topk(k=budget, sorted=True).values[-1]
            # if multiple explained variances lie at threshold it can happen that more PCs are allocated
            remaining_equals_threshold_budget = budget - (all_explained_variance_ratios > threshold).sum()
            # select pcs based on explained variance threshold
            for key in components_as_sd.keys():
                if key.startswith("head."):
                    continue
                # by default pcs are sorted by explained variance ratio -> make sure that this is the case
                explained_variance_ratio = pca[f"{key.replace('.lora_A', '')}.pca.explained_variance_ratio"]
                argsort = explained_variance_ratio.argsort(descending=True)
                assert torch.all(argsort == torch.arange(len(argsort)))
                cur_rank = (explained_variance_ratio > threshold).sum()
                # if explained_variance_ratio == threshold -> use as long as budget is not overdrafted
                if remaining_equals_threshold_budget > 0:
                    num_equals_threshold = (explained_variance_ratio == threshold).sum()
                    if num_equals_threshold > 0:
                        if num_equals_threshold <= remaining_equals_threshold_budget:
                            cur_rank += num_equals_threshold
                            remaining_equals_threshold_budget -= num_equals_threshold
                        else:
                            cur_rank += remaining_equals_threshold_budget
                            remaining_equals_threshold_budget = 0
                components_as_sd[key] = components_as_sd[key][:cur_rank]
            expended_budget = sum([len(value) for key, value in components_as_sd.items() if key.endswith(".lora_A")])
            assert expended_budget == budget, f"expended_budget={expended_budget} budget={budget}"
            # adjust ranks of lora layers
            for key in components_as_sd.keys():
                if key.endswith(".lora_A"):
                    module = model
                    splits = key.split(".")[:-1]
                    if ".".join(splits) in self.names_to_exclude:
                        continue
                    for split in splits:
                        module = getattr(module, split)
                    assert isinstance(module, (LoraLinear, DoraLinear)), f"{key}: {type(module)}"
                    module.change_lora_rank(new_rank=len(components_as_sd[key]))
        else:
            # select highest explained variance ranks if pca rank > lora rank
            for key in components_as_sd.keys():
                if key.startswith("head."):
                    continue
                if len(components_as_sd[key]) > rank:
                    # by default pcs are sorted by explained variance ratio -> make sure that this is the case
                    explained_variance_ratio = pca[f"{key.replace('.lora_A', '')}.pca.explained_variance_ratio"]
                    argsort = explained_variance_ratio.argsort(descending=True)
                    assert torch.all(argsort == torch.arange(len(argsort)))
                    components_as_sd[key] = components_as_sd[key][:rank]
        if self.whiten_pca:
            for key in components_as_sd.keys():
                # get explained_variance_ratio
                assert key.endswith(".lora_A")
                cur_rank = len(components_as_sd[key])
                explained_variance_ratio = pca[f"{key[:-len('.lora_A')]}.pca.explained_variance_ratio"]
                # make sure exp_var is ordered descending
                argsort = explained_variance_ratio.argsort(descending=True)
                assert torch.all(argsort == torch.arange(len(argsort)))
                # whiten
                components_as_sd[key] = components_as_sd[key] / torch.sqrt(explained_variance_ratio)[:cur_rank, None]
        if self.random_rotation:
            for key in components_as_sd.keys():
                assert key.endswith(".lora_A")
                # permute
                cur_rank, dim = components_as_sd[key].shape
                if cur_rank <= 1:
                    continue
                rotation = torch.Tensor(ortho_group.rvs(cur_rank))
                components_as_sd[key] = rotation @ components_as_sd[key].float()
        if self.random_permutation:
            for key in components_as_sd.keys():
                assert key.endswith(".lora_A")
                # permute
                cur_rank = len(components_as_sd[key])
                if cur_rank <= 1:
                    continue
                permut = torch.randn_like(components_as_sd[key]).argsort(dim=0).long()
                components_as_sd[key] = torch.gather(components_as_sd[key], index=permut, dim=0)

        # load lora_a from principal components
        if self.init_with_pcs:
            missing_keys, unexpected_keys = model.load_state_dict(components_as_sd, strict=False)
            # sanity checks
            if len(self.names_to_exclude) == 0:
                assert len(unexpected_keys) == 0, f"unexpected_keys: {unexpected_keys}"
            assert len([key for key in missing_keys if "lora_A" in missing_keys]) == 0

        # freeze everything but lora matrices
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
            else:
                if self.train_biases and "bias" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        # unfreeze head
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True
        return model
