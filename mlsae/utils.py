import functools
import weakref

import torch


def get_repo_id(
    model_name: str,
    expansion_factor: int,
    k: int,
    transformer: bool = False,
    lens: bool = False,
) -> str:
    """
    Get the repo_id that corresponds to the specified hyperparameters.
    You should probably change this!
    """
    model_name = model_name.split("/")[-1]
    repo_id = f"tim-lawson/mlsae-{model_name}-x{expansion_factor}-k{k}"
    if transformer:
        repo_id += "-tfm"
    if lens:
        repo_id += "-lens"
    return repo_id


def get_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def normalize(x: torch.Tensor, dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / torch.max(norm, eps * torch.ones_like(norm))


# Copied from https://stackoverflow.com/a/33672499/23543959
def cache_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator
