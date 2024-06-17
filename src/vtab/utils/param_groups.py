def create_param_groups(model, lr, weight_decay=0., layerwise_lr_decay=None):
    param_groups = []
    for name, param in model.named_parameters():
        properties = {}

        # exlcude from weight decay
        if name.endswith(".bias"):
            properties["weight_decay"] = 0.
        # timm does it like this...not sure if other parameters can also have ndim <= 1
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py
        elif param.ndim <= 1:
            properties["weight_decay"] = 0.
        else:
            properties["weight_decay"] = weight_decay

        # layerwise_lr_decay
        if layerwise_lr_decay is not None:
            properties.update(_get_layerwise_lr_decay_properties(model=model, name=name, decay=layerwise_lr_decay))

        # add param_group
        properties["name"] = name
        properties["lr"] = lr
        properties["params"] = [param]
        param_groups.append(properties)
    return param_groups


def _get_layerwise_lr_decay_properties(model, name, decay):
    # adapted from BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    # this will split the model into len(blocks) + 2 "layers"
    # stem (patch_embed, cls_token, pos_embed) -> blocks -> last norm
    # this means that the last block will already be decayed
    if hasattr(model, "blocks"):
        num_layers = len(model.blocks) + 1
    elif hasattr(model, "model"):
        # e.g. torch_hub_model
        assert hasattr(model.model, "blocks")
        num_layers = len(model.model.blocks) + 1
        if name.startswith("model."):
            name = name[len("model."):]
    else:
        raise NotImplementedError
    scales = list(decay ** (num_layers - i) for i in range(num_layers))
    if (
            name.startswith("patch_embed")
            or name.startswith("cls_token")
            or name.startswith("pos_embed")
            or name.startswith("embed_norm")
            or name == "mask_token"
    ):
        return dict(lr_scale=scales[0])
    elif name.startswith("block"):
        layer = int(name.split('.')[1]) + 1
        return dict(lr_scale=scales[layer])
    elif name.startswith("norm.") or name.startswith("legacy_norm."):
        # last norm is not scaled (i.e. original learning rate)
        return {}
    elif name.startswith("head_norm."):
        # last norm is not scaled (i.e. original learning rate)
        return {}
    elif name.startswith("head."):
        # head is not scaled (i.e. original learning rate)
        return {}
    else:
        raise NotImplementedError
