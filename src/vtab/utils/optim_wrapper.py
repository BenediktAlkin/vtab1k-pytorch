class OptimWrapper:
    def __init__(self, optim, schedule=None):
        super().__init__()
        self.optim = optim
        self.schedule = schedule
        for param_group in self.optim.param_groups:
            assert "original_lr" not in param_group
            param_group["original_lr"] = param_group["lr"]

    def zero_grad(self, set_to_none=True):
        self.optim.zero_grad(set_to_none=set_to_none)

    def step(self, update=None, grad_scaler=None):
        # set lr according to schedule and lr_scale
        if self.schedule is not None:
            assert update is not None
            for param_group in self.optim.param_groups:
                # scale by current schedule progress
                cur_lr = param_group["original_lr"] * self.schedule[update]
                # scale by static scale (e.g. for layer-wise lr decay)
                if "lr_scale" in param_group:
                    cur_lr = cur_lr * param_group["lr_scale"]
                param_group["lr"] = cur_lr

        # step
        if grad_scaler is None:
            self.optim.step()
        else:
            grad_scaler.step(self.optim)
            grad_scaler.update()
