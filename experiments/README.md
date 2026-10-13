# experiments

`experiments` is the campaign orchestration layer for this code release. It connects the local `csi-*` packages to YAML configs and reference job wrappers.

## Included Workflows

- clean training and evaluation;
- white-box, transfer, and UAP robustness studies;
- adversarial training and randomized smoothing;
- campaign job wrappers for the four active study groups.

## Directory Layout

- `attacks_raw_csi/` holds the active robustness entrypoints.
- `defenses_raw_csi/` holds the active defense training and evaluation entrypoints.
- `config/campaign/robustness/` contains the campaign attack configs.
- `config/campaign/defenses/` contains the campaign defense configs.
- `job_scripts/campaign/` contains the reference SGE array launchers and parameter grids.
- `train_models/` contains the clean-model training and Optuna-related runners that feed the campaign.

## Usage Notes

- Install the local `csi-data`, `csi-models`, `csi-attacks`, and `csi-defenses` packages in editable mode before running the experiment modules.
- Dataset roots, checkpoint locations, and result directories are controlled through YAML and environment variables rather than fixed absolute paths.
- This artifact intentionally omits archived analyses, cluster logs, raw datasets, pretrained weights, and bulky result trees.
