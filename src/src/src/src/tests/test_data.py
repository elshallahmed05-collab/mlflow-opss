
from src.data import make_dataset


def test_dataset_shape_and_columns():
    df = make_dataset(n_samples=100)

    assert len(df) == 100
    assert "churn" in df.columns
    assert set(df["churn"].unique()).issubset({0, 1})


def test_dataset_is_deterministic():
    first = make_dataset()
    second = make_dataset()

    assert first.equals(second)
