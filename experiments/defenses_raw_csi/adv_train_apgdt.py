from typing import Optional

from .adv_train_apgd import main as _run_adv_train


def main(config_name: Optional[str] = None) -> None:
    _run_adv_train(
        config_name=config_name,
        attack_method_override="apgdt",
        default_config_relpath="experiments/config/adv_trained/adv_train_apgdt_hid.yaml",
    )


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
