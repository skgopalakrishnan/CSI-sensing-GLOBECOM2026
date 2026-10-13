from typing import Optional

from .transfer_attack_apgd import main as _run_attack


def main(config_name: Optional[str] = None) -> None:
    _run_attack(
        config_name=config_name,
        attack_method_override="apgdt",
        default_attack_type="targeted",
        default_task_family="hid",
        default_free_targeted=True,
        default_config_relpath="experiments/config/robustness/transfer/transfer_targeted_apgdt_hid.yaml",
    )


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
