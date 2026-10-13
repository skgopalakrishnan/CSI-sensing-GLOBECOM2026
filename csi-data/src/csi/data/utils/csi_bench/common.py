import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from torch.utils.data import DataLoader


def resolve_loader_name(split_name):
    if split_name == "train_id":
        return "train"
    if split_name == "val_id":
        return "val"
    if split_name == "test_id":
        return "test"
    if split_name.startswith("test_"):
        return split_name
    return f"test_{split_name}"


def print_dir_contents(path, indent=0):
    print(f"\n==== Directory Contents ({path}) ====")
    try:
        items = os.listdir(path)
        for item in items:
            item_path = os.path.join(path, item)
            print(' ' * indent + (f"- {item}/" if os.path.isdir(item_path) else f"- {item}"))
            if os.path.isdir(item_path):
                print_dir_contents(item_path, indent + 2)
    except Exception as e:
        print(' ' * indent + f"Cannot list contents of {path}: {e}")
    print("================================================\n")


def verify_contents_of_directory(path):
    """
    Utility function to print the contents of a directory for debugging purposes.
    Primarily to check whether metadata and label mapping files exist.

    Args:
        path (str): Path to the directory to be inspected.
    """
    print_dir_contents(path)


def resolve_metadata_path(task_dir, allow_alternates=True):
    metadata_path = os.path.join(task_dir, "metadata", "sample_metadata.csv")
    if os.path.exists(metadata_path):
        return metadata_path
    if not allow_alternates:
        return None

    alternate_metadata_paths = [
        os.path.join(task_dir, "metadata", "metadata.csv"),
        os.path.join(task_dir, "metadata", "sample_metadata.csv"),
        os.path.join(task_dir, "subset_metadata.csv"),
        os.path.join(task_dir, "metadata.csv"),
    ]

    for alt_path in alternate_metadata_paths:
        if os.path.exists(alt_path):
            return alt_path

    return None


def require_metadata_path(task_dir, allow_alternates=True, error_message=None):
    metadata_path = resolve_metadata_path(task_dir, allow_alternates=allow_alternates)
    if metadata_path is None:
        if error_message is None:
            error_message = (
                f"Metadata file not found: "
                f"{os.path.join(task_dir, 'metadata', 'sample_metadata.csv')}"
            )
        raise FileNotFoundError(error_message)
    return metadata_path


def resolve_split_path(task_dir, split_name, allow_alternates=True, splits_dir_name="splits"):
    split_path = os.path.join(task_dir, splits_dir_name, f"{split_name}.json")
    if os.path.exists(split_path):
        return split_path

    if not allow_alternates:
        return None

    # Try alternate locations for split file
    alternate_split_paths = [
        os.path.join(task_dir, f"{splits_dir_name}/{split_name}.json"),
        os.path.join(task_dir, f"{split_name}.json"),
    ]

    for alt_path in alternate_split_paths:
        if os.path.exists(alt_path):
            return alt_path

    return None


def require_splits_path(
    task_dir,
    split_name,
    allow_alternates=True,
    error_message=None,
    splits_dir_name="splits",
):
    split_path = resolve_split_path(
        task_dir,
        split_name,
        allow_alternates=allow_alternates,
        splits_dir_name=splits_dir_name,
    )
    
    if split_path is not None:
        return split_path
    else:
        # List available splits
        splits_dir = os.path.join(task_dir, splits_dir_name)
        available_splits = []
        if os.path.exists(splits_dir) and os.path.isdir(splits_dir):
            for file in os.listdir(splits_dir):
                if file.endswith('.json'):
                    available_splits.append(file.replace('.json', ''))
        if error_message is None:
            error_message = (f"Split file not found: "
                             f"{os.path.join(task_dir, splits_dir_name, f'{split_name}.json')}")      
        raise FileNotFoundError(error_message)
