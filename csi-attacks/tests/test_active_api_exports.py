import importlib


def test_evasion_exports_only_active_attacks():
    evasion = importlib.import_module("csi.attacks.evasion")
    assert hasattr(evasion, "APGD")
    assert hasattr(evasion, "APGDT")
    assert hasattr(evasion, "Attack")
    assert not hasattr(evasion, "PGDL2")
    assert not hasattr(evasion, "PGDL2MMD_Corr")


def test_top_level_package_import_is_resilient():
    attacks_pkg = importlib.import_module("csi.attacks")
    assert hasattr(attacks_pkg, "evasion")
    assert hasattr(attacks_pkg, "utils")
