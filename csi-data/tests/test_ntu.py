import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")
from scipy.io import savemat


def _make_ntu_root(tmp_path, classes=("cls",), per_class=1, modal="CSIamp"):
    root = tmp_path
    rng = np.random.default_rng(0)
    for cls in classes:
        cdir = root / cls
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            # Create data with shape [C, S, T0] where T0//4 = 500 after ::4 slicing
            x = rng.random((3, 114, 2000), dtype=np.float32)
            savemat(str(cdir / f"sample{i}.mat"), {modal: x})
    return root


def test_ntu_shapes_and_norm(tmp_path):
    from csi.data import NTU_Dataset
    root = _make_ntu_root(tmp_path)
    ds = NTU_Dataset(str(root), normalisation="zscore")
    assert len(ds) == 1
    x, y = ds[0]
    assert tuple(x.shape) == (3, 114, 500)
    assert int(y) == 0
    m = float(x.mean())
    assert abs(m) < 1e-2


def _save_mat_2d(path, array, modal="CSIamp"):
    from scipy.io import savemat
    savemat(str(path), {modal: array})


@pytest.mark.parametrize(
    "shape",
    [
        (342, 2000),   # (C*S, T)
        (2000, 342),   # (T, C*S)
        (342, 600),    # non-multiple of 500, should truncate
        (342, 300),    # shorter than 500, should pad
    ],
)
def test_ntu_2d_input_shapes(tmp_path, shape):
    from csi.data import NTU_Dataset
    root = tmp_path
    cdir = root / "cls"
    cdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    x = rng.random(shape, dtype=np.float32)
    _save_mat_2d(cdir / "sample0.mat", x)

    ds = NTU_Dataset(str(root), normalisation="minmax")
    xx, yy = ds[0]
    assert tuple(xx.shape) == (3, 114, 500)
    assert int(yy) == 0


def test_ntu_cleaning_moving_average(tmp_path):
    from csi.data import NTU_Dataset
    root = _make_ntu_root(tmp_path, per_class=1)
    ds_plain = NTU_Dataset(str(root), normalisation="zscore", smoothing=None)
    ds_filtered = NTU_Dataset(
        str(root), normalisation="zscore", smoothing="moving_average", smoothing_kwargs={"window": 9}
    )

    x_plain, _ = ds_plain[0]
    x_filtered, _ = ds_filtered[0]

    plain_var = np.mean(np.abs(np.diff(x_plain.numpy(), axis=-1)))
    filt_var = np.mean(np.abs(np.diff(x_filtered.numpy(), axis=-1)))
    assert filt_var <= plain_var
