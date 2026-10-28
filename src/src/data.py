
"""Generate a deterministic synthetic customer churn dataset."""

import numpy as np
import pandas as pd


def make_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 85, n_samples)
    tenure = rng.integers(0, 73, n_samples)
    charges = rng.uniform(20, 130, n_samples).round(2)
    tickets = rng.poisson(1.5, n_samples).clip(0, 10)

    contract = rng.choice(
        ["month_to_month", "one_year", "two_year"],
        n_samples,
        p=[0.55, 0.25, 0.20],
    )

    internet = rng.choice(
        ["fiber", "dsl", "none"],
        n_samples,
        p=[0.50, 0.35, 0.15],
    )

    logit = (
        -2.0
        + (contract == "month_to_month") * 1.1
        + (tenure < 12) * 0.9
        + (charges > 85) * 0.55
        + tickets * 0.28
        + (internet == "fiber") * 0.25
    )

    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    return pd.DataFrame({
        "age": age,
        "tenure_months": tenure,
        "monthly_charges": charges,
        "support_tickets": tickets,
        "contract_type": contract,
        "internet_service": internet,
        "churn": churn,
    })
