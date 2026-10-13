import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import csv
from ..utils.csi_bench.common import require_metadata_path, require_splits_path
from ..utils.csi_bench.label_utils import load_or_create_label_mapper


class CSI_Bench_Dataset(Dataset):
    """
    Dataset for WiFi CSI benchmark with split handling and metadata loading. Starting with only Fall Detection.
    Supports only HDF5 formats.
    Outputs tensors in (C, F, T) with per-subcarrier normalisation when enabled.
    """
    ## Setting defaults
    # No transforms or label/target transforms.
    # Data column is 'file_path', label column is 'label'
    # Data key in H5 file is 'CSI_amps'
    # Normalisation is 'zscore'
    # No label mapper, no task_dir provided.
    # Debug is True and skip_errors is False (I want to see the info.)
    
    def __init__(self, 
                 dataset_root,
                 task_name, 
                 split_name,
                 splits_dir_name="splits",
                 normalisation="zscore",
                 target_subcarriers=232,
                 target_timesteps=500,
                 transform=None,
                 target_transform=None,
                 channel_axis=None,
                 verbose=False,
                ):
        
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.split_name = split_name
        self.splits_dir_name = splits_dir_name
        self.data_column = "file_path"
        self.label_column = "label"
        self.file_format = "h5"  # Assuming HDF5 format for now
        self.data_key = "CSI_amps"
        self.transform = transform
        self.target_transform = target_transform
        self.channel_axis = channel_axis
        self.verbose = bool(verbose)
        
        ## Input validation for normalisation
        if normalisation not in {"none", "zscore", "minmax"}:
            raise ValueError("normalisation must be one of {'none', 'zscore', 'minmax'}.")
        self.normalisation = normalisation
        
        ## Set task directory and check if metadata and splits exist. Errors will be raise if not found
        self.task_dir = os.path.join(dataset_root, task_name)
        self.metadata_path = require_metadata_path(self.task_dir, allow_alternates=False) 
        self.split_path = require_splits_path(
            self.task_dir,
            split_name,
            splits_dir_name=self.splits_dir_name,
        )      
        
        ## Load split IDs
        with open(self.split_path, 'r') as f:
            self.split_ids = set(json.load(f))
        
        ## Load metadata
        self.metadata = pd.read_csv(self.metadata_path)
        
        ## Filter metadata (rows) for this split
        id_column = 'id' if 'id' in self.metadata.columns else 'sample_id'
        self.split_metadata = self.metadata[self.metadata[id_column].isin(self.split_ids)].reset_index(drop=True)
        
        ## Check if we have the required columns
        if self.data_column not in self.split_metadata.columns:
            raise ValueError(f"Data column '{self.data_column}' not found in metadata")
        if self.label_column not in self.split_metadata.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in metadata")
            
        self._log(f"Loaded {len(self.split_metadata)} samples for {self.task_name} - {self.split_name}")
        
        ## Initialize label mapper if not provided
        mapping_path = os.path.join(self.task_dir, 'metadata', 'label_mapping.json')
        self.label_mapper = load_or_create_label_mapper(
            metadata_path=self.metadata_path,
            label_column=self.label_column,
            mapping_path=mapping_path,
            log_loaded=self.verbose,
        )
                
        ## Set number of classes
        self.num_classes = self.label_mapper.num_classes

        ## Ascertain type of paths stored in metadata
        self.is_path_absolute = self.ascertain_path_type()  # Error if paths are invalid

        ## Set CSI data dimensions (channels inferred from data if possible)
        self.f = int(target_subcarriers)
        self.t = int(target_timesteps)
        self.raw_shape = self._infer_raw_shape()
        self.inferred_channel_axis = self._infer_channel_axis_from_shape(self.raw_shape)
        self.c = self._infer_channels_from_sample()

        ## Missing file logging
        self.missing_log_path = os.path.join(self.task_dir, "metadata", "missing_files.csv")
        self._missing_logged = set()
        self._load_missing_log()

    def _load_missing_log(self):
        if not os.path.exists(self.missing_log_path):
            return
        try:
            with open(self.missing_log_path, "r", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        self._missing_logged.add(row[0])
        except Exception:
            pass

    def _log_missing_path(self, path: str):
        if path in self._missing_logged:
            return
        self._missing_logged.add(path)
        os.makedirs(os.path.dirname(self.missing_log_path), exist_ok=True)
        try:
            with open(self.missing_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([path])
        except Exception:
            pass
        
    def ascertain_path_type(self):
        """This module determines the type of path provided in the metadata and resolves it to an absolute path."""
        
        ## Get the first file path from metadata to analyze
        sample_path = self.split_metadata.iloc[0][self.data_column]
        self._log(f"Sample file path from metadata: {sample_path}")
        
        if os.path.isabs(sample_path):
            self._log("Detected absolute file paths in metadata.")
            # Check if the absolute path exists
            if sample_path.startswith('E:/'):
                raise NotImplementedError(
                    "Absolute paths starting with 'E:/' are not supported. "
                    "Please modify the metadata to use relative paths."
                )
            elif os.path.exists(sample_path):
                self._log("Absolute paths are valid and exist.")
                return True
        else:
            self._log("Detected relative file paths in metadata.")
            # Check if relative to task directory
            task_relative = os.path.join(self.task_dir, sample_path)
            if os.path.exists(task_relative):
                self._log("Relative paths are valid with respect to task directory.")
                return False
            else:
                self._log("Relative paths are NOT valid with respect to task directory.")
                raise FileNotFoundError(
                    f"Path is relative: {task_relative}.\n"
                    "But it cannot be found in the task directory."
                )
        return False

    def _infer_channel_axis_from_shape(self, shape):
        if shape is None:
            return None
        if len(shape) == 2:
            return None
        if len(shape) != 3:
            raise ValueError(f"Unsupported CSI sample shape for channel-axis inference: {shape}")
        if self.channel_axis is not None:
            axis = int(self.channel_axis)
            if axis < 0:
                axis += 3
            if axis < 0 or axis >= 3:
                raise ValueError(f"channel_axis={self.channel_axis} is invalid for shape {shape}.")
            return axis

        # Pick axis where remaining two dimensions best match (F, T), allowing swapped ordering.
        best_axis = 0
        best_err = None
        for axis in range(3):
            dims = [int(shape[i]) for i in range(3) if i != axis]
            err = min(
                abs(dims[0] - self.f) + abs(dims[1] - self.t),
                abs(dims[0] - self.t) + abs(dims[1] - self.f),
            )
            if best_err is None or err < best_err or (err == best_err and int(shape[axis]) < int(shape[best_axis])):
                best_axis = axis
                best_err = err
        return best_axis

    def _infer_channels_from_sample(self):
        """Infer channel count from the first sample if possible."""
        if self.raw_shape is not None:
            shape = self.raw_shape
        else:
            sample_path = self.split_metadata.iloc[0][self.data_column]
            if self.is_path_absolute:
                filepath = sample_path
            else:
                filepath = os.path.join(self.task_dir, sample_path)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found for channel inference: {filepath}")
            data = self._read_h5_file(filepath)
            self._log(f"Sample data shape for channel inference: {data.shape if data is not None else 'None'}")
            if data is None:
                raise ValueError("No data found in sample for channel count inference.")
            shape = data.shape

        if len(shape) == 2:
            return 1
        if len(shape) == 3:
            axis = self._infer_channel_axis_from_shape(shape)
            return int(shape[axis])
        raise ValueError(f"Unsupported CSI sample shape for channel inference: {shape}")

    def _infer_raw_shape(self):
        """Infer raw CSI shape from the first sample before resizing."""
        sample_path = self.split_metadata.iloc[0][self.data_column]
        if self.is_path_absolute:
            filepath = sample_path
        else:
            filepath = os.path.join(self.task_dir, sample_path)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found for raw shape inference: {filepath}")
        data = self._read_h5_file(filepath)
        shape = data.shape if data is not None else None
        self._log(f"Sample raw data shape: {shape if shape is not None else 'None'}")
        return shape

    def _read_h5_file(self, filepath: str) -> np.ndarray | None:
        """Read CSI data from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            if len(f.keys()) == 0:
                self._log(f"Skipping file {filepath} due to error: Empty H5 file")
                return None
            if self.data_key in f:
                return np.array(f[self.data_key])
            if 'csi' in f:
                return np.array(f['csi'])
            if 'CSI' in f:
                return np.array(f['CSI'])
            self._log(
                f"Skipping file {filepath} due to error: Could not find data in H5 file. "
                f"Available keys: {list(f.keys())}"
            )
            return None

    def _to_cft(self, csi_data: np.ndarray) -> np.ndarray:
        csi_data = np.asarray(csi_data, dtype=np.float32)
        if csi_data.ndim == 2:
            resized = self._resample_2d(csi_data, self.f, self.t)
            return resized[np.newaxis, ...]
        if csi_data.ndim != 3:
            raise ValueError(f"Unsupported CSI sample shape: {csi_data.shape}")

        axis = self._infer_channel_axis_from_shape(csi_data.shape)
        arranged = np.moveaxis(csi_data, axis, 0)

        # Resolve whether remaining axes are (F, T) or (T, F).
        err_ft = abs(arranged.shape[1] - self.f) + abs(arranged.shape[2] - self.t)
        err_tf = abs(arranged.shape[1] - self.t) + abs(arranged.shape[2] - self.f)
        if err_tf < err_ft:
            arranged = arranged.transpose(0, 2, 1)

        if arranged.shape[1] == self.f and arranged.shape[2] == self.t:
            return arranged.astype(np.float32, copy=False)

        out = np.empty((arranged.shape[0], self.f, self.t), dtype=np.float32)
        for idx in range(arranged.shape[0]):
            out[idx] = self._resample_2d(arranged[idx], self.f, self.t)
        return out
    
    def __len__(self):
        return len(self.split_metadata)    
    
    def __getitem__(self, idx):
        """Get sample by index"""
        try:
            row = self.split_metadata.iloc[idx]
            
            ## Get file path
            filepath = row[self.data_column]

            if self.is_path_absolute:
                if not os.path.exists(filepath):
                    self._log(f"Initial file path does not exist: {filepath}")
                    self._log_missing_path(filepath)
                    return None
                else:
                    filepath = filepath  # Use as is
            elif not self.is_path_absolute:
                if not os.path.exists(os.path.join(self.task_dir, filepath)):
                    missing_path = os.path.join(self.task_dir, filepath)
                    self._log(f"Initial file path does not exist: {missing_path}")
                    self._log_missing_path(missing_path)
                    return None
                else:
                    filepath = os.path.join(self.task_dir, filepath)  # Relative to task dir
            else:  # This section is theoretically unreachable
                raise RuntimeError("Unreachable code reached in path resolution.")
            
            ## Load CSI data from HDF5 file
            csi_data = None
            try:
                csi_data = self._read_h5_file(filepath)
            except Exception as e:
                self._log(f"Skipping file {filepath} due to error: '{str(e)}'")
            
            ## Check if data is valid
            if csi_data is None or csi_data.size == 0:
                self._log(f"Skipping file {filepath} due to error: Empty data")
                return None
            ## Preprocess CSI data
            if csi_data.ndim == 3 and csi_data.shape[-1] == 1:
                csi_data = np.squeeze(csi_data, axis=-1)
            csi_data = self._to_cft(csi_data)

            if self.normalisation == "zscore":
                if isinstance(csi_data, torch.Tensor):
                    csi_data = csi_data.detach().cpu().numpy()
                mean = csi_data.mean(axis=-1, keepdims=True, dtype=np.float32)
                std = csi_data.std(axis=-1, keepdims=True, dtype=np.float32)
                csi_data = ((csi_data - mean) / (std + 1e-6)).astype(np.float32, copy=False)
            elif self.normalisation == "minmax":
                if isinstance(csi_data, torch.Tensor):
                    csi_data = csi_data.detach().cpu().numpy()
                min_val = csi_data.min(axis=-1, keepdims=True)
                max_val = csi_data.max(axis=-1, keepdims=True)
                denom = max_val - min_val
                csi_data = np.where(denom > 1e-6, (csi_data - min_val) / denom, csi_data - min_val)

            if not isinstance(csi_data, torch.Tensor):
                csi_data = torch.from_numpy(np.asarray(csi_data, dtype=np.float32)).float()
            
            ## Get labels
            label = row[self.label_column]
            
            ## Convert string label to integer using mapper
            label_idx = self.label_mapper.transform(label)
            
            if self.transform is not None:
                csi_data = self.transform(csi_data)
            if self.target_transform is not None:
                label_idx = self.target_transform(label_idx)
            return csi_data, label_idx
        
        except Exception as e:
            self._log(f"Error processing sample {idx}: {str(e)}")
            return None  

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def _resample_2d(array: np.ndarray, target_f: int, target_t: int) -> np.ndarray:
        """Resample a 2D (F,T) array to (target_f, target_t) using numpy interpolation."""
        if array.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {array.shape}")
        f_in, t_in = array.shape
        if f_in == target_f and t_in == target_t:
            return array.astype(np.float32, copy=False)
        t_base = np.arange(t_in, dtype=np.float32)
        t_idx = np.linspace(0, t_in - 1, target_t, dtype=np.float32)
        temp = np.empty((f_in, target_t), dtype=np.float32)
        for fi in range(f_in):
            temp[fi] = np.interp(t_idx, t_base, array[fi].astype(np.float32, copy=False))
        f_base = np.arange(f_in, dtype=np.float32)
        f_idx = np.linspace(0, f_in - 1, target_f, dtype=np.float32)
        out = np.empty((target_f, target_t), dtype=np.float32)
        for ti in range(target_t):
            out[:, ti] = np.interp(f_idx, f_base, temp[:, ti])
        return out
