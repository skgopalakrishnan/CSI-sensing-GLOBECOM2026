from typing import Optional

from .uap_attack_apgd import main as _run_attack


def main(config_name: Optional[str] = None) -> None:
    _run_attack(
        config_name=config_name,
        attack_method_override="apgd",
        default_attack_type="untargeted",
        default_task_family="har",
        default_free_targeted=False,
        default_config_relpath="experiments/config/robustness/uap/untargeted_uap_apgd_har.yaml",
    )


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
