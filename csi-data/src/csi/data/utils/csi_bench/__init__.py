__all__ = [
    "load_benchmark_supervised",
    "load_contrastive_benchmark_ssl",
    "load_contrastive_benchmark_ssl_loaders",
    "load_classification_benchmark_ssl",
    "load_classification_with_val_benchmark_ssl",
    "load_augmented_train_benchmark_ssl",
    "load_mae_benchmark_ssl",
]


def load_benchmark_supervised(*args, **kwargs):
    from .loaders import load_benchmark_supervised as _impl

    return _impl(*args, **kwargs)


def load_contrastive_benchmark_ssl(*args, **kwargs):
    from .loaders import load_contrastive_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def load_contrastive_benchmark_ssl_loaders(*args, **kwargs):
    from .loaders import load_contrastive_benchmark_ssl_loaders as _impl

    return _impl(*args, **kwargs)


def load_classification_benchmark_ssl(*args, **kwargs):
    from .loaders import load_classification_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def load_classification_with_val_benchmark_ssl(*args, **kwargs):
    from .loaders import load_classification_with_val_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def load_augmented_train_benchmark_ssl(*args, **kwargs):
    from .loaders import load_augmented_train_benchmark_ssl as _impl

    return _impl(*args, **kwargs)


def load_mae_benchmark_ssl(*args, **kwargs):
    from .loaders import load_mae_benchmark_ssl as _impl

    return _impl(*args, **kwargs)
