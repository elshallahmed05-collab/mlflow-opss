
"""Train a churn prediction model and register it with MLflow."""

import os
import time

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import make_dataset


MODEL_NAME = os.getenv("MODEL_NAME", "ChurnClassifier")


def train():
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("customer-churn")

    df = make_dataset()
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    numeric_features = [
        "age",
        "tenure_months",
        "monthly_charges",
        "support_tickets",
    ]
    categorical_features = [
        "contract_type",
        "internet_service",
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=120,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ])

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        mlflow.log_params({
            "model": "RandomForestClassifier",
            "n_estimators": 120,
            "random_state": 42,
        })

        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, predictions),
            "f1": f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
        })

        signature = infer_signature(
            X_train,
            model.predict(X_train),
        )

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
            registered_model_name=MODEL_NAME,
        )

        print("Training completed!")
        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    for attempt in range(30):
        try:
            train()
            break
        except Exception:
            if attempt == 29:
                raise
            print("Waiting for MLflow server...")
            time.sleep(2)
