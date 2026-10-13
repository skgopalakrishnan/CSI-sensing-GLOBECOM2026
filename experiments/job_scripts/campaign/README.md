# Campaign Job Scripts

This directory contains the reference SGE array wrappers for the four active campaign studies.

## Scripts

- `run_study1_whitebox_array.sh`
- `run_study2_transfer_array.sh`
- `run_study3_uap_array.sh`
- `run_study4_defense_train_array.sh`
- `run_study4_defense_eval_array.sh`

## Notes

- Each launcher reads one line from the matching file in `params/` using `$SGE_TASK_ID`.
- The parameter files define dataset, model, seed, and config coverage for each study.
- Result paths are assembled from `CSI_RESULTS_JOB_DIR`, `CSI_RESULTS_RUN_PARTS`, and the resolved config path.
- These scripts document the original cluster workflow, but they are included as reference helpers rather than a promise of turnkey execution on every system.
