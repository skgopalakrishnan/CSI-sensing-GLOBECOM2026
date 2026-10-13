import json
import os
from pathlib import Path
from typing import Union, Optional, Mapping, Any, Dict, List, Tuple

import torch
from torch.utils.data import Subset

from .datasets import get_dataset_spec


def _infer_head_type_from_model_name(model_name: str) -> Optional[str]:
    """Best-effort inference of head type ('tcn', 'gru', 'ssm') from model name."""
    name = model_name.strip().lower()
    for suffix in ("_tcn", "_gru", "_ssm"):
        if name.endswith(suffix):
            return suffix[1:]
    # Handle case without leading underscore (e.g. 'aeheadtcn')
    for suffix in ("tcn", "gru", "ssm"):
        if name.endswith(suffix):
            return suffix
    return None


def _normalise_dataset_key(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def get_weights_path(dataset_name: str, model_name: str, root: str, model_folder_name: str, variant: Optional[str] = None) -> Optional[str]:
    # Save/Load convention (default): <root>/<dataset>/<model_folder>/<dataset>_<model>.pt
    # With variant: <model_folder>_<variant>
    dataset_name = dataset_name.strip()
    dataset_key = dataset_name.lower().replace(" ", "_")
    if dataset_key in {"csi-bench", "csi_bench"}:
        dataset_name = "CSI-Bench"
    model_name = model_name.strip()
    folder = dataset_name

    # if model_folder_name is already the relative path, use it directly
    folder_name = os.path.expanduser(model_folder_name.strip())
    if variant:
        folder_name = f"{folder_name}_{variant}"

    is_rel_path = any(sep in folder_name for sep in (os.sep, '/', '\\'))  # relative path if it contains a separa

    if os.path.isabs(folder_name):
        base_dir = folder_name
    elif is_rel_path:
        dataset_base = os.path.join(root, folder, folder_name)
        cwd_base = os.path.normpath(os.path.join(os.getcwd(), folder_name))
        if os.path.isdir(dataset_base) or not os.path.isdir(cwd_base):
            base_dir = dataset_base
        else:
            base_dir = cwd_base
    else:
        base_dir = os.path.join(root, folder, folder_name)

    def _epoch_candidates(prefix: str, dir_path: Path) -> List[Path]:
        """Return candidate checkpoint files ordered by preference."""
        candidates: List[Path] = []
        preferred_epochs: Tuple[int, ...] = (20, 15, 10)
        for epoch in preferred_epochs:
            cand = dir_path / f"{prefix}_epoch{epoch:02d}.pt"
            if cand.exists():
                candidates.append(cand)
                return candidates
        def _epoch_key(path: Path) -> int:
            try:
                return int(path.stem.split("_epoch")[-1])
            except Exception:
                return -1
        globbed = sorted(
            dir_path.glob(f"{prefix}_epoch*.pt"),
            key=_epoch_key,
            reverse=True,
        )
        candidates.extend(globbed)
        return candidates

    dir_path = Path(base_dir)
    prefix = f"{dataset_name}_{model_name}"
    full_path = dir_path / f"{prefix}.pt"

    if full_path.exists():
        return str(full_path)

    for cand in _epoch_candidates(prefix, dir_path):
        if cand.exists():
            return str(cand)

    # Fallback: scan for metadata-backed checkpoints with matching dataset/head
    base_path = Path(base_dir)
    if base_path.is_dir():
        expected_head = _infer_head_type_from_model_name(model_name)
        dataset_key = _normalise_dataset_key(dataset_name)
        matched_path: Optional[str] = None
        for candidate in sorted(base_path.glob("*.pt")):
            meta_path = candidate.with_suffix(candidate.suffix + ".meta.json")
            meta: Dict[str, Any] = {}
            if meta_path.exists():
                try:
                    with meta_path.open("r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except Exception:
                    meta = {}
            meta_dataset = _normalise_dataset_key(str(meta.get("dataset", "")))
            if meta_dataset != dataset_key:
                continue
            if expected_head is not None:
                meta_head = str(meta.get("head", {}).get("type", "")).strip().lower()
                if meta_head and meta_head != expected_head:
                    continue
            matched_path = str(candidate)
            break
        if matched_path is not None:
            return matched_path

    raise FileNotFoundError(
        f"Model weights not found for {dataset_name}/{model_name} in {base_dir}. "
        f"Searched for '{prefix}.pt' and epoch-suffixed variants."
    )


def load_weighted_model(model, path, device: Union[str, torch.device]='cuda', show_params=False):
    # Handle both string and torch.device objects
    if isinstance(device, torch.device):
        device_str = device.type
        device_obj = device
    else:
        device_str = device
        device_obj = torch.device(device)
    
    if os.path.exists(path):
        # Accept both bare state_dict checkpoints and wrapped payloads with metadata.
        payload = torch.load(path, map_location='cpu')
        state_dict = payload
        if isinstance(payload, Mapping):
            if "model_state_dict" in payload and isinstance(payload["model_state_dict"], Mapping):
                state_dict = payload["model_state_dict"]
            elif "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
                state_dict = payload["state_dict"]
        model.load_state_dict(state_dict)
        model.to(device_obj)
        model.eval()
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        
        if model_device.type != device_obj.type:
            print(f"WARNING: Model device mismatch! Expected {device_obj}, got {model_device}")
            # Force move again
            model.to(device_obj)
            print(f"Model forcibly moved to {device_obj}")
            
        if show_params:
            print(f"Model architecture:")
            print(model)
            print(f"Model parameters:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: {param.size()}")        
    else:
        print(f"Model weights file not found at {path}")
    return model


def get_num_classes(
    dataset: str,
    *,
    task_name: Optional[str] = None,
    arch_metadata: Optional[Mapping[str, Any]] = None,
) -> int:
    if arch_metadata is not None and arch_metadata.get("num_classes") is not None:
        try:
            return int(arch_metadata["num_classes"])
        except Exception as exc:
            raise ValueError("arch_metadata['num_classes'] must be an integer") from exc

    try:
        spec = get_dataset_spec(dataset, task_name=task_name, metadata=arch_metadata)
    except ValueError:
        spec = None

    if spec is not None and spec.num_classes is not None:
        return int(spec.num_classes)

    mapping = {
        "NTU-Fi-HumanID": 14,
        "NTU-Fi_HAR": 6,
    }
    if dataset in mapping:
        return mapping[dataset]

    raise ValueError(
        f"Unknown number of classes for dataset '{dataset}'. "
        "Pass `num_classes=` or include `arch_metadata['num_classes']`."
    )


def infer_num_classes(dataset) -> Optional[int]:
    seen = set()
    current = dataset
    while isinstance(current, Subset):
        if id(current) in seen:
            break
        seen.add(id(current))
        current = current.dataset
    for candidate in (dataset, current):
        if candidate is None:
            continue
        num_classes = getattr(candidate, "num_classes", None)
        if num_classes is not None:
            return int(num_classes)
        labels = getattr(candidate, "labels", None)
        if labels is not None:
            if len(labels) == 0:
                continue
            return int(max(labels)) + 1
    return None


def load_arch_metadata(weights_path: Union[str, os.PathLike[str]]) -> Optional[Mapping[str, Any]]:
    """Load the architecture metadata stored alongside a checkpoint, if present."""
    path = Path(weights_path)
    meta_path = Path(f"{path}.meta.json")
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _normalise_model_name(name: str) -> str:
    return name.lower().replace(" ", "").replace("-", "")
