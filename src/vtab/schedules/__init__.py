def create_schedule(kind, total_updates):
    if kind.startswith("warmup-cosine-"):
        warmup_duration_identifier = kind[len("warmup-cosine-"):]
        if warmup_duration_identifier[-1] == "%":
            # percentage based (e.g. warmup-cosine-10%)
            warmup_updates = int(total_updates / int(warmup_duration_identifier[:-1]))
        else:
            raise NotImplementedError
        from .warmup_cosine_schedule import WarmupCosineSchedule
        return WarmupCosineSchedule(warmup_updates=warmup_updates, total_updates=total_updates)
    if kind.startswith("warmup-linear-"):
        warmup_duration_identifier = kind[len("warmup-linear-"):]
        if warmup_duration_identifier[-1] == "%":
            # percentage based (e.g. warmup-linear-10%)
            warmup_updates = int(total_updates / int(warmup_duration_identifier[:-1]))
        else:
            raise NotImplementedError
        from .warmup_linear_schedule import WarmupLinearSchedule
        return WarmupLinearSchedule(warmup_updates=warmup_updates, total_updates=total_updates)
    raise NotImplementedError(f"invalid schedule kind: '{kind}'")
