# csi-defenses

`csi-defenses` contains the countermeasures used by the active CSI robustness campaigns in this artifact.

## Install

`pip install -e ./csi-defenses`

## Includes

- shared loss modules for robust optimization;
- adversarial training routines such as `PGD-AT` and `TRADES`;
- randomized smoothing training and certification helpers;
- utilities consumed by the `experiments/defenses_raw_csi` entrypoints.

Use this package when you want the defense-side training and certification code without the surrounding experiment wrappers.
