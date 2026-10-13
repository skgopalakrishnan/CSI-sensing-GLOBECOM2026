from typing import Optional

from .randomized_smoothing import main as _run_rs


def main(config_name: Optional[str] = None) -> None:
    _run_rs(
        config_name=config_name,
        default_config_relpath="experiments/config/campaign/defenses/randomized_smoothing/rs_har.yaml",
    )


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
