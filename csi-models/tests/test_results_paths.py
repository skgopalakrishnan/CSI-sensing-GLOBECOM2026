import importlib.util
from pathlib import Path


RESULTS_PATH = Path(__file__).resolve().parents[1] / "src" / "csi" / "models" / "utils" / "results.py"
RESULTS_SPEC = importlib.util.spec_from_file_location("csi_models_results", RESULTS_PATH)
assert RESULTS_SPEC is not None and RESULTS_SPEC.loader is not None
RESULTS_MODULE = importlib.util.module_from_spec(RESULTS_SPEC)
RESULTS_SPEC.loader.exec_module(RESULTS_MODULE)
prepare_results_dir = RESULTS_MODULE.prepare_results_dir


def test_prepare_results_dir_appends_job_run_parts_from_env(monkeypatch, tmp_path):
    job_dir = tmp_path / "study1_whitebox_12345"
    cfg_path = tmp_path / "study1_cfg.yaml"
    cfg_path.write_text("common:\n  experiment_name: demo\n", encoding="utf-8")

    monkeypatch.setenv("CSI_RESULTS_JOB_DIR", str(job_dir))
    monkeypatch.setenv(
        "CSI_RESULTS_RUN_PARTS",
        "task_19/seed_2/dataset_csi_bench/model_CSITimeSformer/run_demo_cfg_123",
    )

    out_dir, _ts, _cfg_name = prepare_results_dir(
        experiment_name="campaign_whitebox_har_constrained",
        cfg_path=str(cfg_path),
        mode="untargeted_apgd",
        relative_parts=[],
    )

    out_path = Path(out_dir)
    assert out_path == (
        job_dir
        / "campaign_whitebox_har_constrained"
        / "untargeted_apgd"
        / "task_19"
        / "seed_2"
        / "dataset_csi_bench"
        / "model_CSITimeSformer"
        / "run_demo_cfg_123"
    )
    assert out_path.is_dir()


def test_prepare_results_dir_preserves_relative_parts_and_appends_job_run_parts(monkeypatch, tmp_path):
    job_dir = tmp_path / "study2_transfer_12345"
    cfg_path = tmp_path / "study2_cfg.yaml"
    cfg_path.write_text("common:\n  experiment_name: demo\n", encoding="utf-8")

    monkeypatch.setenv("CSI_RESULTS_JOB_DIR", str(job_dir))
    monkeypatch.setenv("CSI_RESULTS_RUN_PARTS", "task_79/dataset_ntu/source_model_MLP/target_model_LeNet")

    out_dir, _ts, _cfg_name = prepare_results_dir(
        experiment_name="campaign_transfer_har_constrained",
        cfg_path=str(cfg_path),
        mode="transfer_untargeted_apgd",
        relative_parts=["apgd", "untargeted", "har", "seed_2"],
    )

    out_path = Path(out_dir)
    assert out_path == (
        job_dir
        / "campaign_transfer_har_constrained"
        / "transfer_untargeted_apgd"
        / "apgd"
        / "untargeted"
        / "har"
        / "seed_2"
        / "task_79"
        / "dataset_ntu"
        / "source_model_MLP"
        / "target_model_LeNet"
    )
    assert out_path.is_dir()
