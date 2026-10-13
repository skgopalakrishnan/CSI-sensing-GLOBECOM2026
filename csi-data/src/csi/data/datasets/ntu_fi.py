import glob
import os
from functools import partial
from typing import Callable, Mapping, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from torch import Tensor, from_numpy, stack
from torch.utils.data import Dataset
import torch

from .transform import TransformDataset as TD


class NTU_Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        modal: str = "CSIamp",
        dtype: str = "tensor",
        n_items: int = 100,
        normalisation: str = "none",
        smoothing: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = None,
        smoothing_kwargs: Optional[Mapping[str, Union[int, float, str]]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.modal = modal
        self.dtype = dtype
        self.n_items = n_items
        if normalisation not in {"minmax", "zscore", "none"}:
            raise ValueError("normalisation must be one of {'minmax', 'zscore', 'none'}.")
        self.normalisation = normalisation
        self.smoothing = smoothing
        self.smoothing_kwargs = dict(smoothing_kwargs or {})
        self._smoothing_fn = self._resolve_smoothing(smoothing, self.smoothing_kwargs)
        
        # Specify nominal shape
        self.c = 3      # channels
        self.f = 114    # subcarriers
        self.t = 500    # timesteps
        
        self.samples = []
        self.samples_by_class = {}
        classes = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        self.class_to_idx = {p.name: i for i, p in enumerate(classes)}
        for cdir in classes:
            label = self.class_to_idx[cdir.name]
            for mat_file in sorted(cdir.glob("*.mat")):  # <-- sort files for consistency
                path = str(mat_file)
                self.samples.append((path, label))
                self.samples_by_class.setdefault(label, []).append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple:
        if isinstance(idx, slice):  # Dataloader does not use this by the way
            indices = list(range(*idx.indices(len(self))))
            xs, ys = [], []
            for i in indices:
                path, label = self.samples[i]
                x = self._read_and_normalise(path)
                xs.append(x)
                ys.append(label)
            return self._output_in_type(xs, ys)
        path, label = self.samples[idx]
        x = self._read_and_normalise(path)
        if self.dtype == "tensor":
            return x, torch.as_tensor(label, dtype=torch.long)
        x_np = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float32)
        return x_np, np.int64(label)
    
    def _output_in_type(self, xs, ys):
        if self.dtype == "tensor":
            xs = stack(xs, dim=0)
            ys = Tensor(ys).long()
        elif self.dtype in ("npy", "numpy"):
            xs = np.stack([x.numpy() if isinstance(x, Tensor) else x for x in xs], axis=0)
            ys = np.array(ys, dtype=np.int64)
        else:
            raise ValueError("Unsupported dtype. Use 'tensor' or 'numpy'.")
        return xs, ys
    
    def _read_and_normalise(self, path) -> Tensor:
        mat_data = loadmat(path, squeeze_me=True, struct_as_record=False)
        x = mat_data[self.modal]
        
        # Ensure we operate on float32 for downstream filters
        x = np.asarray(x, dtype=np.float32)
        
        # Ensure NTU shape
        x = TD._ensure_cft(x, self.c, self.f, self.t)

        # Apply smoothing per (C,F) over time
        if self._smoothing_fn is not None:
            x = self._smoothing_fn(x)  # expects same shape back
        
        # Normalisation (per subcarrier always)
        if self.normalisation == 'zscore':
            mu  = x.mean(axis=-1, keepdims=True, dtype=np.float64)
            sd  = x.std(axis=-1, keepdims=True,  dtype=np.float64)
            x = ((x - mu) / (sd + 1e-6)).astype(np.float32, copy=False)
        elif self.normalisation == 'minmax':
            xmin = x.min(axis=-1, keepdims=True)
            xmax = x.max(axis=-1, keepdims=True)
            denom = (xmax - xmin)
            x = np.where(denom > 1e-8, (x - xmin) / denom, x - xmin).astype(np.float32, copy=False)
        else:
            pass  # no normalisation

        return from_numpy(x).float()

    # def retrieve_samples(self):
    #     if not self.samples:
    #         return self._output_in_type([], [])
    #     all_samples = []
    #     for label, class_samples in self.samples_by_class.items():
    #         n_to_sample = min(len(class_samples), self.n_items)
    #         indices = np.random.choice(len(class_samples), n_to_sample, replace=False)
    #         all_samples.extend([class_samples[i] for i in indices])
    #     xs, ys = [], []
    #     for path, label in all_samples:
    #         x = self._read_and_normalize(path)
    #         xs.append(x)
    #         ys.append(label)
    #     return self._output_in_type(xs, ys)

    # ------------------------------------------------------------------
    # Smoothing helpers
    # ------------------------------------------------------------------
    def _resolve_smoothing(
        self,
        smoothing: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]],
        params: Mapping[str, Union[int, float, str]],
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        if smoothing is None:
            return None

        if callable(smoothing):
            fn = partial(smoothing, **params) if params else smoothing
                
            def _custom(arr: np.ndarray) -> np.ndarray:
                result = np.asarray(fn(arr), dtype=np.float32)
                if result.shape != arr.shape:
                    raise ValueError(f"Custom smoothing changed shape {arr.shape} -> {result.shape}")
                return result
            return _custom  # picklable if 'smoothing' is at top level
    
        if isinstance(smoothing, str):
            key = smoothing.lower()
            if key in {"moving_average", "moving-average", "ma", "mean"}:
                window = int(params.get("window", 5))
                return partial(__class__._moving_average_filter, window=window)
            if key in {"median", "medfilt"}:
                return self._median_filter_factory(params)
            if key in {"savgol", "savitzky_golay", "savitzky-golay"}:
                return self._savgol_filter_factory(params)
            raise ValueError(f"Unknown smoothing method '{smoothing}' for NTU_Dataset.")
        raise TypeError("smoothing must be None, a callable, or a recognised string identifier.")


    @staticmethod
    def _moving_average_filter(arr: np.ndarray, window: int) -> np.ndarray:
        window = int(max(1, window))
        if window <= 1:
            return arr
        # enforce odd window for symmetric smoothing
        if window % 2 == 0:
            window += 1
            
        try:
            from scipy.ndimage import uniform_filter1d
        except ImportError as exc:  # pragma: no cover - depends on SciPy extras
            raise ImportError(
                "Moving average filtering requires scipy.ndimage.uniform_filter1d. Please install SciPy."
            ) from exc
            
        if arr.ndim != 3:
            raise ValueError(f"Moving average expects 3D (C,F,T), got {arr.shape}")
        
        out = uniform_filter1d(arr, size=window, axis=-1, mode="nearest")
        return out.astype(arr.dtype, copy=False)


    def _median_filter_factory(
        self, params: Mapping[str, Union[int, float, str]]
    ) -> Callable[[np.ndarray], np.ndarray]:
        try:
            from scipy.ndimage import median_filter
        except ImportError as exc:  # pragma: no cover - depends on SciPy extras
            raise ImportError(
                "Median filtering requires scipy.ndimage.median_filter. Please install SciPy."
            ) from exc

        window = int(params.get("window", 5))
        if window % 2 == 0:
            window += 1
        kernel = (1, 1, max(1, window))

        def _median(arr: np.ndarray) -> np.ndarray:
            if arr.ndim != 3:
                raise ValueError(f"Median expects 3D (C,F,T), got {arr.shape}")
            out = median_filter(arr, size=kernel, mode="nearest")
            return out.astype(arr.dtype, copy=False)

        return _median   


    def _savgol_filter_factory(
        self, params: Mapping[str, Union[int, float, str]]
    ) -> Callable[[np.ndarray], np.ndarray]:
        try:
            from scipy.signal import savgol_filter
        except ImportError as exc:  # pragma: no cover - depends on SciPy extras
            raise ImportError(
                "Savitzky-Golay filtering requires scipy.signal.savgol_filter. Please install SciPy."
            ) from exc

        window = int(params.get("window", 11))
        poly = int(params.get("polyorder", 3))
        mode = str(params.get("mode", "mirror"))
        
        # ensure minimally valid window, then ensure odd
        window = max(window, poly + 2)
        if window % 2 == 0:
            window += 1
        if not (poly < window):
            raise ValueError(f"polyorder ({poly}) must be < window ({window}).")
        
        def _savgol(arr: np.ndarray) -> np.ndarray:
            if arr.ndim != 3:
                raise ValueError(f"SavGol expects 3D (C,F,T), got {arr.shape}")
            out = savgol_filter(arr, window_length=window, polyorder=poly, 
                                axis=-1, mode=mode)
            return out.astype(arr.dtype, copy=False)
        return _savgol
