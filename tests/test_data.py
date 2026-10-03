from pathlib import Path


def test_project_data_directories_exist():
    assert Path("data/raw").exists()
    assert Path("data/processed").exists()
    assert Path("data/drift").exists()
