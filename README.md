# CSI-sensing

`CSI-sensing` is the code release for our WiFi Channel State Information (CSI) robustness campaigns. It contains the package code and campaign scripts used in the paper, without the private runtime state that lived in the original research workspace.

This code corresponds to the paper:

> **Towards Trustworthy Wi-Fi CSI-based Sensing: Systematic Evaluation of Adversarial Robustness**
> Shreevanth Krishnaa Gopalakrishnan and Stephen Hailes, Department of Computer Science, University College London.
> *2026 IEEE Globecom Workshops (GC Wkshps), 6th Workshop on Machine Learning and Deep Learning for Wireless Security*, Macau, December 2026.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{gopalakrishnan2026trustworthy,
  author    = {Gopalakrishnan, Shreevanth Krishnaa and Hailes, Stephen},
  title     = {Towards Trustworthy {Wi-Fi} {CSI}-based Sensing: Systematic Evaluation of Adversarial Robustness},
  booktitle = {2026 IEEE Globecom Workshops (GC Wkshps)},
  address   = {Macau},
  year      = {2026}
}
```

## Scope

- Included: `csi-data`, `csi-models`, `csi-attacks`, `csi-defenses`, `experiments/`, and dataset directory stubs under `datasets/`.
- Excluded: raw datasets, pretrained weights, archived analyses, cluster logs, large result dumps, and unrelated workspace automation.
- Intended use: installable, importable, and inspectable reference implementation rather than a one-command reproduction bundle.

## Repository Layout

- `csi-data` provides dataset readers, preprocessing helpers, and loader builders for `NTU-Fi_HAR`, `NTU-Fi-HumanID`, and `CSI-Bench`.
- `csi-models` provides model factories, training utilities, metrics, and result helpers for the campaign backbones.
- `csi-attacks` provides the adversarial attack implementations used by the robustness studies.
- `csi-defenses` provides adversarial training and randomized smoothing utilities.
- `experiments` contains the YAML configs, Python entrypoints, and campaign job wrappers that connect the packages.
- `datasets` contains only directory stubs that document the expected on-disk layout for the supported datasets.

## Datasets

The datasets are not redistributed here. Obtain them from their original sources and follow their terms of use:

- **NTU-Fi HAR** and **NTU-Fi HumanID**: distributed with the SenseFi benchmark, <https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark>. J. Yang et al., "SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing," *Patterns*, 4(3), 2023, doi:10.1016/j.patter.2023.100703.
- **CSI-Bench**: G. Zhu et al., "CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing," *NeurIPS* 2025, doi:10.52202/085713-5648.

## Environment

- Developed and checked with Python `3.11.8` on Linux.
- The original campaign runs used multi-job Linux environments with NVIDIA GPUs.

## Quick Start

1. Create a Python environment and install the shared requirements:
   `pip install -r requirements.txt`
2. Install the local packages in editable mode:
   `pip install -e ./csi-data -e ./csi-models -e ./csi-attacks -e ./csi-defenses`
3. Place dataset contents under `datasets/NTU-Fi_HAR`, `datasets/NTU-Fi-HumanID`, and `datasets/CSI-Bench` using the checked-in folder stubs as guides.
4. Run the package tests, e.g. `pytest csi-models/tests csi-attacks/tests`.
5. Train clean models with the entrypoints under `experiments/train_models/` and the configs under `experiments/config/training/`.
6. Use the templates under `experiments/config/campaign/` and the launch helpers under `experiments/job_scripts/campaign/` as the campaign reference workflow.

## Notes

- Dataset paths and checkpoint locations are config-driven rather than hardcoded.
- If your data lives on a share mounted at different paths on different machines, set `CSI_STORAGE_ROOT` and `CSI_STORAGE_ROOT_ALT` to the two mount points and absolute paths are remapped between them. The training job scripts also honour `CSI_VENV` (virtual environment directory) and `CSI_PYTHON_SOURCE` (optional shell file to source before activation).
- No raw CSI corpora or trained weights are bundled with this release.
- The job scripts target a Sun Grid Engine (SGE) cluster. The campaign wrappers source site-specific helpers (`experiments/job_scripts/common_job_env.sh`, `experiments/job_scripts/prepare_campaign_config.py`) that are not included; they are provided as reference helpers rather than a guarantee of turnkey execution on arbitrary systems.

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
