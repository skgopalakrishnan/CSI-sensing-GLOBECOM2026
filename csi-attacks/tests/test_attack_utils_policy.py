import pytest

from csi.attacks.utils.attack_utils import (
    build_label_mapping,
    canonical_project_name,
    enforce_har_hid_attack_policy,
)


def test_canonical_project_name_aliases():
    assert canonical_project_name("har") == "HumanActivityRecognition"
    assert canonical_project_name("human_activity_recognition") == "HumanActivityRecognition"
    assert canonical_project_name("hid") == "HumanIdentification"
    assert canonical_project_name("NTU-Fi_HAR") == "NTU-Fi_HAR"


def test_build_label_mapping_with_explicit_num_classes():
    mapping_u, targeted_u = build_label_mapping("HumanActivityRecognition", "untargeted", num_classes=4)
    assert targeted_u is False
    assert mapping_u == {0: [0], 1: [1], 2: [2], 3: [3]}

    mapping_t, targeted_t = build_label_mapping("HumanIdentification", "targeted", num_classes=4)
    assert targeted_t is True
    assert mapping_t == {0: [1], 1: [2], 2: [3], 3: [0]}


def test_build_label_mapping_requires_known_or_explicit_classes():
    with pytest.raises(ValueError, match="Unknown project"):
        build_label_mapping("UnknownTask", "untargeted")


@pytest.mark.parametrize(
    "dataset_name,attack_type,attack_method",
    [
        ("HumanActivityRecognition", "untargeted", "apgd"),
        ("HumanIdentification", "targeted", "apgd"),
        ("HumanIdentification", "targeted", "apgdt"),
        ("NTU-Fi_HAR", "untargeted", "apgd"),
        ("NTU-Fi-HumanID", "targeted", "apgd"),
        ("NTU-Fi-HumanID", "targeted", "apgdt"),
    ],
)
def test_enforce_har_hid_policy_accepts_valid_combinations(dataset_name, attack_type, attack_method):
    attack_mode, method = enforce_har_hid_attack_policy(dataset_name, attack_type, attack_method, strict=True)
    assert attack_mode == attack_type
    assert method == attack_method


def test_enforce_har_hid_policy_rejects_invalid_combinations():
    with pytest.raises(ValueError, match="HAR experiments must use attack.type='untargeted'"):
        enforce_har_hid_attack_policy("HumanActivityRecognition", "targeted", "apgdt", strict=True)

    with pytest.raises(ValueError, match="HID experiments must use attack.type='targeted'"):
        enforce_har_hid_attack_policy("HumanIdentification", "untargeted", "apgdt", strict=True)

    with pytest.raises(ValueError, match=r"HID targeted experiments must use attack.method in \{'apgd','apgdt'\}"):
        enforce_har_hid_attack_policy("HumanIdentification", "targeted", "fab", strict=True)
