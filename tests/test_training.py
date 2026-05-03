# tests/test_training.py

from src.data import load_data, split_data
from src.features import build_preprocessing_pipeline


def test_load_data():
    df = load_data()

    assert df is not None
    assert df.shape[0] > 0
    assert "MEDV" in df.columns


def test_split_data():
    df = load_data()
    X, y = split_data(df)

    assert X.shape[0] == y.shape[0]
    assert "MEDV" not in X.columns


def test_build_preprocessing_pipeline():
    df = load_data()
    X, _ = split_data(df)

    preprocessor = build_preprocessing_pipeline(X)

    assert preprocessor is not None