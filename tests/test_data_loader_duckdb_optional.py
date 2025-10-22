import importlib
import importlib.util
import sys

import pandas as pd
import pytest


@pytest.fixture()
def restore_data_loader():
    original_module = importlib.import_module("services.data_loader")
    yield original_module
    importlib.reload(original_module)


def test_load_dataset_without_duckdb(monkeypatch, tmp_path, restore_data_loader):
    original_module = restore_data_loader

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "duckdb":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    sys.modules.pop("services.data_loader", None)
    module = importlib.import_module("services.data_loader")

    data = pd.DataFrame(
        {
            "集計年": [2020, 2021],
            "産業大分類コード": ["A", "A"],
            "産業大分類名": ["テスト産業", "テスト産業"],
            "業種中分類コード": ["B1", "B1"],
            "業種中分類名": ["テスト業種", "テスト業種"],
            "指標1": ["1.0", "2.0"],
        }
    )
    csv_path = tmp_path / "sample.csv"
    data.to_csv(csv_path, index=False)

    result = module.load_dataset(csv_path, tmp_path / "app.duckdb")
    expected = module.preprocess_dataframe(data)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

    sys.modules["services.data_loader"] = original_module
