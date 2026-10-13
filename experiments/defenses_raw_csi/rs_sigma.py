"""Sigma resolution helpers for randomized smoothing experiment configs."""

from __future__ import annotations

from typing import Any, List, Mapping


def resolve_sigma_for_domain(rs_cfg: Mapping[str, Any], *, domain: str) -> float:
    # Preferred behavior: resolve sigma by domain + SNR from global calibration map.
    by_domain_snr = rs_cfg.get("sigma_global_by_domain_snr_db") or rs_cfg.get("sigma_by_domain_snr_db") or {}
    if isinstance(by_domain_snr, Mapping) and by_domain_snr:
        domain_map = by_domain_snr.get(domain)
        if not isinstance(domain_map, Mapping):
            raise ValueError(
                f"sigma_global_by_domain_snr_db is configured but missing domain '{domain}'. "
                f"Available domains: {sorted(str(k) for k in by_domain_snr.keys())}."
            )

        snr_raw = rs_cfg.get("snr_db")
        if isinstance(snr_raw, (list, tuple)):
            snr_raw = snr_raw[0] if snr_raw else None
        if snr_raw is None:
            raise ValueError(
                "sigma_global_by_domain_snr_db requires certified_robustness.randomized_smoothing.snr_db."
            )

        snr_candidates: List[str] = [str(snr_raw).strip()]
        try:
            snr_candidates.append(f"{float(snr_raw):g}")
        except Exception:
            pass

        for key in snr_candidates:
            if key in domain_map and domain_map[key] is not None:
                sigma = float(domain_map[key])
                if sigma <= 0.0:
                    raise ValueError(f"sigma map for domain={domain} snr={key} must be > 0.")
                return sigma

        raise ValueError(
            f"Could not resolve sigma for domain='{domain}' at snr_db={snr_raw!r}. "
            f"Available SNR keys: {sorted(str(k) for k in domain_map.keys())}."
        )

    # Legacy fallbacks (only used when global-by-domain map is absent).
    by_domain = rs_cfg.get("sigma_by_domain") or {}
    if isinstance(by_domain, Mapping):
        sigma_dom = by_domain.get(domain)
        if sigma_dom is not None:
            sigma = float(sigma_dom)
            if sigma <= 0.0:
                raise ValueError(f"sigma_by_domain.{domain} must be > 0.")
            return sigma

    sigma_raw = rs_cfg.get("sigma")
    if sigma_raw is not None:
        sigma = float(sigma_raw)
        if sigma <= 0.0:
            raise ValueError("certified_robustness.randomized_smoothing.sigma must be > 0.")
        return sigma

    raise ValueError(
        "Could not resolve smoothing sigma. Set one of: "
        "sigma, sigma_by_domain.<domain>, or sigma_global_by_domain_snr_db.<domain>.<snr_db>."
    )
