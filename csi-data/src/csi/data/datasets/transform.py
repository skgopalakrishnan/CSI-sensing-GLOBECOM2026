"""Dataset wrapper that applies a transform to each sample."""
from typing import Callable, Optional, Tuple, Mapping, Union, Any
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import Tensor
from torchvision import transforms
import random

@dataclass
class ReconstructionAugment:
    amplitude_jitter: float = 0.0
    gaussian_std: float = 0.0
    subcarrier_dropout: float = 0.0
    time_dropout: float = 0.0
    random_mask: float = 0.0
    mask_value: float = 0.0


class TransformDataset(Dataset):
    """Dataset wrapper applying transforms and normalisation."""
    def __init__(self, 
                 base: Dataset, 
                 base_normalisation: str = "none",
                 task: str = "reconstruction",
                 transform: Optional[Callable[[Tensor], Tensor]] = None,
                 augment: Optional[Mapping[str, float]] = None,
                 ) -> None:
                
        self.base = base
        self.normalisation = (base_normalisation or "").lower()
        self.transform = transform    
        aug = augment or {}
        self.augment = ReconstructionAugment(
            amplitude_jitter=float(aug.get("amplitude_jitter", 0.0)),
            gaussian_std=float(aug.get("gaussian_std", 0.0)),
            subcarrier_dropout=float(aug.get("subcarrier_dropout", 0.0)),
            time_dropout=float(aug.get("time_dropout", 0.0)),
            random_mask=float(aug.get("mask_fraction", 0.0)),
            mask_value=float(aug.get("mask_value", 0.0)),
        )
        if task not in {"reconstruction", "classification"}:
            self.task = "reconstruction"  # Default to reconstruction
        else:
            self.task = task

    def __len__(self) -> int:
        return len(self.base) # type: ignore

    def __getattr__(self, name: str) -> Union[Callable, Any]:
        # Delegate attribute access to base dataset if not found here
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self.base, name)
    
    def __getitem__(self, index):
        sample, label = self.base[index]
        if not isinstance(sample, Tensor):
            sample = torch.as_tensor(sample)

        sample = sample.float()
        if self.transform is not None:  # Apply macro transforms
            sample = self.transform(sample)

        if self.task == "reconstruction" and self.normalisation is not None:
            target = sample.clone()
            noisy, mask = self._corrupt_sample(sample)
            return noisy, target, mask
        else:
            if not isinstance(label, Tensor):
                label = torch.tensor(label, dtype=torch.long)
            else:
                label = label.long()
            return sample, label
    
    # def _apply_normalisation(self, tensor: Tensor) -> Tensor:
    #     out = tensor
    #     if self.normalization in {"instance", "sample"}:
    #         mean = out.mean()
    #         std = out.std(unbiased=False).clamp_min(1e-5)
    #         out = (out - mean) / std
    #     if self.per_subcarrier or self.normalization == "subcarrier":
    #         out = per_subcarrier_zscore(out)
    #     return out

    def _corrupt_sample(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
        noisy = tensor.clone()
        mask = torch.ones_like(noisy)
        if self.augment.amplitude_jitter > 0:
            magnitude = self.augment.amplitude_jitter
            factor = 1.0 + torch.empty(1).uniform_(-magnitude, magnitude).item()
            noisy = noisy * factor
        if self.augment.gaussian_std > 0:
            noisy = noisy + torch.randn_like(noisy) * self.augment.gaussian_std
        if self.augment.subcarrier_dropout > 0:
            noisy, mask = TransformDataset._frequency_dropout_augment(noisy, mask, self.augment.subcarrier_dropout)
        if self.augment.time_dropout > 0:
            noisy, mask = TransformDataset._time_dropout_augment(noisy, mask, self.augment.time_dropout)
        if self.augment.random_mask > 0:
            noisy, mask = TransformDataset._mask_random_augment(noisy, mask, self.augment.random_mask)
        if self.augment.mask_value != 0.0:
            noisy = torch.where(mask > 0, noisy, torch.full_like(noisy, self.augment.mask_value))
        return noisy, mask

    @staticmethod
    def compose_transforms(*funcs: Optional[Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
        steps = [fn for fn in funcs if fn is not None]
        if not steps:
            return lambda x: x  # Identity
        def _composed(tensor: Tensor) -> Tensor:
            for fn in steps:
                tensor = fn(tensor)
            return tensor
        return _composed
    
    @staticmethod
    def downsample_time_transform(target_len: Optional[int]) -> Optional[Callable[[Tensor], Tensor]]:
        if target_len is None or target_len <= 0:
            return None
        def _transform(tensor: Tensor) -> Tensor:
            # Expect at least (..., F, T)
            if tensor.dim() < 3:
                return tensor
            current = tensor.shape[-1]
            if current == target_len:
                return tensor
            if current > target_len and current % target_len == 0:
                factor = current // target_len
                # Merge all leading dims except the last (time) into one channel axis for pooling
                reshaped = tensor.contiguous().view(-1, 1, current)  # contiguous for view-safety
                pooled = F.avg_pool1d(reshaped, kernel_size=factor, stride=factor)
                return pooled.view(*tensor.shape[:-1], target_len)
            # Otherwise use 2D interpolation over (F, T) treating preceding as channels
            resized = F.interpolate(
                tensor.unsqueeze(0),                # (1, C, F, T) or (1, ..., F, T)
                size=(tensor.shape[-2], target_len),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            return resized

        return _transform
    
    @staticmethod
    def zscore_per_subc_transform(mean: torch.Tensor, std: torch.Tensor):
        def _transform(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() < 3:
                raise ValueError(f"Expected tensor with at least 3 dims, received shape {tuple(tensor.shape)}")
            m = mean.to(tensor.device, tensor.dtype).unsqueeze(-1)
            s = torch.clamp(std.to(tensor.device, tensor.dtype).unsqueeze(-1), min=1e-6)
            return (tensor - m) / s
        return _transform
    
    @staticmethod
    def zscore_dataset_transform(mean: torch.Tensor, std: torch.Tensor):
        def _transform(tensor: torch.Tensor) -> torch.Tensor:
            m = mean.to(tensor.device, tensor.dtype)
            s = torch.clamp(std.to(tensor.device, tensor.dtype), min=1e-6)
            return (tensor - m) / s
        return _transform
    
    @staticmethod
    def create_channel_dim_transform(tensor: Tensor) -> Tensor:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor
    
    @staticmethod
    def swap_time_freq_transform(tensor: Tensor) -> Tensor:
        if tensor.dim() >= 3:
            tensor = tensor.transpose(-2, -1).contiguous()
        return tensor

    @staticmethod
    def _frequency_dropout_augment(tensor: Tensor, mask: Tensor, fraction: float) -> Tuple[Tensor, Tensor]:
        if fraction <= 0.0:
            return tensor, mask
        keep_prob = max(0.0, 1.0 - fraction)
        if keep_prob >= 1.0:
            return tensor, mask
        freq_keep = (torch.rand(tensor.shape[-2], device=tensor.device) < keep_prob).to(tensor.dtype)
        if freq_keep.sum().item() == 0:
            freq_keep = freq_keep.fill_(1.0)
        freq_keep = freq_keep.view(*([1] * (tensor.dim() - 2)), tensor.shape[-2], 1)
        tensor = tensor * freq_keep
        mask = mask * freq_keep
        return tensor, mask

    @staticmethod
    def _time_dropout_augment(tensor: Tensor, mask: Tensor, fraction: float) -> Tuple[Tensor, Tensor]:
        if fraction <= 0.0:
            return tensor, mask
        keep_prob = max(0.0, 1.0 - fraction)
        if keep_prob >= 1.0:
            return tensor, mask
        time_keep = (torch.rand(tensor.shape[-1], device=tensor.device) < keep_prob).to(tensor.dtype)
        if time_keep.sum().item() == 0:
            time_keep = time_keep.fill_(1.0)
        time_keep = time_keep.view(*([1] * (tensor.dim() - 1)), tensor.shape[-1])
        tensor = tensor * time_keep
        mask = mask * time_keep
        return tensor, mask

    @staticmethod
    def _mask_random_augment(tensor: Tensor, mask: Tensor, fraction: float) -> Tuple[Tensor, Tensor]:
        if fraction <= 0.0:
            return tensor, mask
        keep_prob = max(0.0, 1.0 - fraction)
        if keep_prob >= 1.0:
            return tensor, mask
        element_keep = (torch.rand_like(tensor) < keep_prob).to(tensor.dtype)
        if element_keep.sum().item() == 0:
            element_keep.fill_(1.0)
        tensor = tensor * element_keep
        mask = mask * element_keep
        return tensor, mask

    @staticmethod
    def _ensure_cft(x: np.ndarray, c: int, f: int, t: int) -> np.ndarray:
        # Shape to (c,f,t)
        if x.ndim == 2:
            h, w = x.shape
            if h == c*f:      x = x.reshape(c, f, w)
            elif w == c*f:    x = x.T.reshape(c, f, h)
            elif max(h, w) >= t and min(h, w) == c*f:
                x = x.reshape(c,f,w) if h == c*f else x.T.reshape(c,f,h)
            else:
                raise IndexError(f"Unexpected data shape {x.shape}; cannot reshape to ({c},{f},{t}).")
        elif x.ndim == 3:
            if   (x.shape[0], x.shape[1]) == (c,f): pass
            elif (x.shape[0], x.shape[1]) == (f,c): x = x.transpose(1,0,2)
            elif x.shape[0]*x.shape[1] == c*f:      x = x.reshape(c,f,x.shape[2])
            else: raise IndexError(f"Unexpected 3D shape {x.shape}; cannot map to ({c},{f},{t}).")

        # Resample/pad to T=t
        T = x.shape[-1]
        if T == t:
            return x
        if T > 1:
            idx = np.linspace(0, T - 1, t, dtype=np.float32)
            base = np.arange(T, dtype=np.float32)
            flat = x.reshape(-1, T)
            out = np.empty((flat.shape[0], t), dtype=np.float32)
            for i in range(flat.shape[0]):
                out[i] = np.interp(idx, base, flat[i])
            x = out.reshape(c, f, t)
        else:
            # T==1 → repeat
            x = np.repeat(x, t, axis=-1)
        return x


## Self-Supervised Learning Transforms

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

class SkipNoneDataset(Dataset):
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

class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]

class Compose(object):
    def __init__(self, transforms_list):
        self.transforms = list(transforms_list)

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x
    
class ApplyTransformDataset(Dataset):
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
    
class SkipNoneDataset(Dataset):
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