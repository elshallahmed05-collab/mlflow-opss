
from src.data import make_dataset


def test_expected_features_exist():
    df = make_dataset(n_samples=50)

    expected_columns = {
        "age",
        "tenure_months",
        "monthly_charges",
        "support_tickets",
        "contract_type",
        "internet_service",
        "churn",
    }

    assert expected_columns.issubset(df.columns)


def test_dataset_contains_both_target_classes():
    df = make_dataset(n_samples=1200)

    assert set(df["churn"].unique()) == {0, 1}
