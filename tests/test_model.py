from pathlib import Path


def test_model_artifact_path_is_defined():
    assert Path("artifacts/serving_model").exists()
