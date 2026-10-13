import random

import torch
import torch.utils.data as data
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms


############################################################################################################


class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = int(n_views)

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class Compose(object):
    def __init__(self, transforms_list):
        self.transforms = list(transforms_list)

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class RandomResizedCrop2D(object):
    def __init__(self, output_size=None, scale=(0.5, 1.0), ratio=(0.15, 0.15), p=0.5):
        self.output_size = output_size
        self.scale = scale
        self.ratio = ratio
        self.p = float(p)

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        if x.dim() != 3:
            raise ValueError("Expected tensor shape (C, F, T) for RandomResizedCrop2D.")
        _, f_in, t_in = x.shape
        area = f_in * t_in
        for _ in range(10):
            target_area = area * random.uniform(self.scale[0], self.scale[1])
            aspect = random.uniform(self.ratio[0], self.ratio[1])
            crop_f = int(round((target_area * aspect) ** 0.5))
            crop_t = int(round((target_area / aspect) ** 0.5))
            if 0 < crop_f <= f_in and 0 < crop_t <= t_in:
                top = random.randint(0, f_in - crop_f)
                left = random.randint(0, t_in - crop_t)
                cropped = x[:, top : top + crop_f, left : left + crop_t]
                return self._resize(cropped, f_in, t_in)
        return self._resize(x, f_in, t_in)

    def _resize(self, x, f_in, t_in):
        if self.output_size is None:
            out_f, out_t = f_in, t_in
        else:
            out_f, out_t = int(self.output_size[0]), int(self.output_size[1])
        if x.shape[-2] == out_f and x.shape[-1] == out_t:
            return x
        return torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(out_f, out_t),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)


class GaussianBlur2D(object):
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.5):
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        self.p = float(p)

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        return self.blur(x)


class Resize2D(object):
    def __init__(self, output_size=None):
        self.output_size = output_size

    def __call__(self, x):
        if self.output_size is None:
            return x
        if x.dim() != 3:
            raise ValueError("Expected tensor shape (C, F, T) for Resize2D.")
        out_f, out_t = int(self.output_size[0]), int(self.output_size[1])
        if x.shape[-2] == out_f and x.shape[-1] == out_t:
            return x
        return torch.nn.functional.interpolate(
            x.unsqueeze(0),
            size=(out_f, out_t),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)


class AdditiveGaussianNoise(object):
    def __init__(self, std=0.05, p=0.5):
        self.std = float(std)
        self.p = float(p)

    def __call__(self, x):
        if self.std <= 0 or torch.rand(1).item() > self.p:
            return x
        return x + torch.randn_like(x) * self.std


class AmplitudeJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, p=0.5):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.p = float(p)

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x
        out = x
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            out = out * factor
        if self.contrast > 0:
            mean = out.mean(dim=(-2, -1), keepdim=True)
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            out = (out - mean) * factor + mean
        return out


class ApplyTransformDataset(data.Dataset):
    def __init__(self, base_dataset, transform=None, target_transform=None):
        super().__init__()
        self.base_dataset = base_dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if sample is None:
            return None
        if len(sample) != 2:
            raise ValueError("Expected dataset to return (sample, label).")
        x, y = sample
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


class SkipNoneDataset(data.Dataset):
    def __init__(self, base_dataset, max_retries=20):
        super().__init__()
        self.base_dataset = base_dataset
        self.max_retries = int(max_retries)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            sample = self.base_dataset[idx]
            if sample is not None:
                return sample
            idx = random.randint(0, len(self.base_dataset) - 1)
        raise RuntimeError("Failed to sample a valid CSI-Bench item after retries.")


def contrastive_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        raise ValueError("Received an empty batch in contrastive_collate.")
    views, labels = zip(*batch)
    if not isinstance(views[0], (list, tuple)):
        return default_collate(batch)
    n_views = len(views[0])
    for view_set in views:
        if len(view_set) != n_views:
            raise ValueError("Inconsistent number of views in contrastive batch.")
    stacked_views = [default_collate([view_set[i] for view_set in views]) for i in range(n_views)]
    labels = default_collate(list(labels))
    return stacked_views, labels


def filter_none_collate(batch):
    """Drop invalid samples and collate the remaining supervised batch."""
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return torch.zeros(0, 1, 1, 1), torch.zeros(0, dtype=torch.long)
    return default_collate(batch)


def build_contrast_transforms_timeseries(
    output_size=None,
    scale=(0.5, 1.0),
    ratio=(0.15, 0.15),
    blur_kernel=3,
    blur_sigma=(0.1, 2.0),
    noise_std=0.05,
    brightness=0.4,
    contrast=0.4,
    p=0.5,
):
    return Compose([
        RandomResizedCrop2D(output_size=output_size, scale=scale, ratio=ratio, p=p),
        GaussianBlur2D(kernel_size=blur_kernel, sigma=blur_sigma, p=p),
        AdditiveGaussianNoise(std=noise_std, p=p),
        AmplitudeJitter(brightness=brightness, contrast=contrast, p=p),
    ])


def build_classification_transforms_timeseries():
    return Compose([])


def build_mae_transforms_timeseries(output_size=None):
    # Keep MAE target consistency: deterministic spatial transform only.
    return Compose([
        Resize2D(output_size=output_size),
    ])


__all__ = [
    "ContrastiveTransformations",
    "Compose",
    "RandomResizedCrop2D",
    "GaussianBlur2D",
    "Resize2D",
    "AdditiveGaussianNoise",
    "AmplitudeJitter",
    "ApplyTransformDataset",
    "SkipNoneDataset",
    "contrastive_collate",
    "filter_none_collate",
    "build_contrast_transforms_timeseries",
    "build_classification_transforms_timeseries",
    "build_mae_transforms_timeseries",
]
