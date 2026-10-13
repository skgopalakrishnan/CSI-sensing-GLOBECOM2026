from __future__ import annotations

import pytest

from experiments.defenses_raw_csi.rs_sigma import resolve_sigma_for_domain


def test_resolve_sigma_prefers_global_by_domain_snr_over_legacy_overrides() -> None:
    cfg = {
        "snr_db": 20,
        "sigma": 0.3,
        "sigma_by_domain": {"ntu": 0.2},
        "sigma_global_by_domain_snr_db": {
            "ntu": {
                "20": 0.0844,
            }
        },
    }
    sigma = resolve_sigma_for_domain(cfg, domain="ntu")
    assert sigma == pytest.approx(0.0844)


def test_resolve_sigma_accepts_numeric_snr_key_formats() -> None:
    cfg = {
        "snr_db": 20.0,
        "sigma_global_by_domain_snr_db": {
            "csi_bench": {
                "20": 0.0841,
            }
        },
    }
    sigma = resolve_sigma_for_domain(cfg, domain="csi_bench")
    assert sigma == pytest.approx(0.0841)


def test_resolve_sigma_raises_if_global_map_missing_domain() -> None:
    cfg = {
        "snr_db": 20,
        "sigma_global_by_domain_snr_db": {
            "ntu": {
                "20": 0.0844,
            }
        },
    }
    with pytest.raises(ValueError, match="missing domain"):
        resolve_sigma_for_domain(cfg, domain="csi_bench")


def test_resolve_sigma_raises_if_global_map_missing_snr_db() -> None:
    cfg = {
        "sigma_global_by_domain_snr_db": {
            "ntu": {
                "20": 0.0844,
            }
        },
    }
    with pytest.raises(ValueError, match="requires .*snr_db"):
        resolve_sigma_for_domain(cfg, domain="ntu")


def test_resolve_sigma_falls_back_to_legacy_when_global_map_absent() -> None:
    cfg = {
        "sigma_by_domain": {"ntu": 0.1},
        "sigma": 0.2,
    }
    sigma = resolve_sigma_for_domain(cfg, domain="ntu")
    assert sigma == pytest.approx(0.1)

    sigma_global = resolve_sigma_for_domain({"sigma": 0.2}, domain="ntu")
    assert sigma_global == pytest.approx(0.2)
