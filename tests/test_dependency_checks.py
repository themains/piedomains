import builtins
import importlib.util
import pathlib

import pytest

UTILS_PATH = pathlib.Path(__file__).resolve().parents[0] / ".." / "piedomains" / "utils.py"
UTILS_PATH = UTILS_PATH.resolve()

spec = importlib.util.spec_from_file_location("pd_utils", UTILS_PATH)
pd_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd_utils)
safe_import_pandas = pd_utils.safe_import_pandas


def test_safe_import_pandas_success():
    pd = safe_import_pandas()
    df = pd.DataFrame({"a": [1]})
    assert df.loc[0, "a"] == 1


def test_safe_import_pandas_failure(monkeypatch):
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pandas":
            raise ValueError("numpy.dtype size changed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError):
        safe_import_pandas()
