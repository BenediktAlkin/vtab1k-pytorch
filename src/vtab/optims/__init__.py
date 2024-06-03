def create_optim(optim, model, lr, batch_size, schedule=None):
    # scale lr by batchsize
    lr_scale_factor = optim.pop("lr_scale_factor")
    if batch_size != lr_scale_factor:
        scaled_lr = lr * 64 / batch_size
    else:
        scaled_lr = lr

    # create param groups
    from vtab.utils.param_groups import create_param_groups
    param_groups = create_param_groups(
        model=model,
        lr=scaled_lr,
        weight_decay=optim.pop("weight_decay", 0.),
        layerwise_lr_decay=optim.pop("layerwise_lr_decay", None),
    )

    # create torch optim
    from vtab.utils.factory import instantiate
    optim = instantiate(
        module_name="torch.optim",
        type_name=optim.pop("kind"),
        params=param_groups,
        **optim,
    )
    from vtab.utils.optim_wrapper import OptimWrapper
    return OptimWrapper(optim=optim, schedule=schedule)
