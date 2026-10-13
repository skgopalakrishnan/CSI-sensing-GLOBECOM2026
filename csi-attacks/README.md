# csi-attacks

`csi-attacks` contains the active adversarial attack code used by the campaign studies in this artifact.

## Install

`pip install -e ./csi-attacks`

## Active Scope

- Maintained evasion attacks: `APGD`, `APGDT`
- Legacy attack families are not part of the active paper workflow in this snapshot

## Public Entry Points

- `csi.attacks.evasion.attack_factory.create_attack(...)`
- `csi.attacks.evasion.attack_factory.generate_X_adv(...)`
- `csi.attacks.evaluation.evaluate_pgdl2_over_budgets(...)`
- `csi.attacks.evaluation.evaluate_asr_pgdl2_untargeted(...)`
- `csi.attacks.utils.attack_utils.build_label_mapping(...)`

## Notes

- HAR campaign runs default to untargeted attacks.
- HID campaign runs default to targeted attacks, with `APGDT` used for the white-box HID sweep.
- Visualization and model-complexity helpers rely on optional extras such as `matplotlib` and `ptflops`.
