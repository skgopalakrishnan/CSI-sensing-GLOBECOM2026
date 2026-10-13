# csi-models

`csi-models` contains the model definitions and training utilities used by the campaign scripts in this artifact.

## Install

`pip install -e ./csi-models`

## Includes

- supervised CSI backbones and lightweight AE-head variants;
- reusable training loops, metrics, seeding, and result helpers;
- config-loading utilities used by the `experiments/` entrypoints.

## Notes

- `PyYAML` is a core dependency because the package exposes config-loading utilities.
- Plotting and some statistical summary helpers are optional extras rather than hard requirements for every workflow.
