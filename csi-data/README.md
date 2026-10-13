# csi-data

`csi-data` provides the dataset readers and preprocessing utilities used by the CSI robustness campaigns in this artifact.

## Install

`pip install -e ./csi-data`

## Public API

- Loaders: `from csi.data import get_experiment_loaders`
- Datasets: `from csi.data import NTU_Dataset, CSI_Bench_Dataset`

## Included Dataset Families

- `NTU-Fi_HAR`
- `NTU-Fi-HumanID`
- `CSI-Bench`

## Notes

- Dataset paths are supplied through experiment YAML rather than hardcoded inside the package.
- The package uses the shared `csi` PEP 420 namespace so it can coexist with `csi.models`, `csi.attacks`, and `csi.defenses`.
- Optional EDA helpers live under `csi.data.utils.common.eda_utils` and rely on the visualization-oriented packages listed in the repository root `requirements.txt`.
