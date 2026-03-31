from dataset.imagenet import build_imagenet_code
from dataset.t2i import build_t2i


def build_dataset(args, **kwargs):
    if args.dataset == "imagenet_code":
        return build_imagenet_code(args, **kwargs)
    if args.dataset == "t2i":
        return build_t2i(args, **kwargs)

    raise ValueError(f"dataset {args.dataset} is not supported in EvoAR-v1")
