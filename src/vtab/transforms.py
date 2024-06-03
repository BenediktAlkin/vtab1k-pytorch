def create_transform(transform):
    if transform is None:
        return None
    from torchvision.transforms import Compose, ToTensor, Normalize, InterpolationMode, RandomResizedCrop
    if transform == "resize224bicubic_imagenetnorm":
        from torchvision.transforms import CenterCrop, Resize
        return Compose(
            [
                Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
    if transform == "resize224bicubic_hflip_imagenetnorm":
        from torchvision.transforms import CenterCrop, Resize, RandomHorizontalFlip
        return Compose(
            [
                Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
    if transform == "rrc008_imagenetnorm":
        return Compose(
            [
                RandomResizedCrop(size=224, interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
    if transform == "rrc02_imagenetnorm":
        return Compose(
            [
                RandomResizedCrop(size=224, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
    if transform == "rrc08_imagenetnorm":
        return Compose(
            [
                RandomResizedCrop(size=224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
    raise NotImplementedError(f"invalid transform '{transform}'")
