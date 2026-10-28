# MLflow Ops — ready-to-run starter

A complete local MLOps demo: reproducible scikit-learn training, MLflow experiment tracking and model registry, FastAPI prediction service, Docker Compose, tests, and GitHub Actions CI.

## Requirements
- Docker Desktop / Docker Engine + Compose plugin
- Or Python 3.11+ to run without Docker

## Quick start (Docker)
```bash
docker compose up --build
```
Then open:
- MLflow UI: http://localhost:5000
- Prediction API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

The `trainer` service runs once and logs a model to MLflow. The API loads the latest version in the `ChurnClassifier` registered model. On first startup, API waits briefly for model registration; if startup races with training, restart the API:
```bash
docker compose restart api
```

## Quick start (Python, no Docker)
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```
In a second terminal (activate venv, same folder):
```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m src.train
```
Windows PowerShell equivalent:
```powershell
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
python -m src.train
```
Start API in another terminal:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Set `MODEL_NAME=ChurnClassifier` if using a different registry model name.

## Test a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":42,"tenure_months":18,"monthly_charges":79.5,"support_tickets":2,"contract_type":"month_to_month","internet_service":"fiber"}'
```
Example response:
```json
{"prediction":1,"label":"churn","probability":0.83}
```
Probability will vary with the fitted model.

## Run tests
```bash
pytest -q
```

## Architecture
- `src/data.py`: deterministic synthetic churn dataset for a self-contained demo.
- `src/train.py`: split, preprocessing, train, metrics, MLflow logging, model registration.
- `src/api.py`: loads the latest registered model and exposes `/predict` and `/health`.
- `tests/`: unit tests and API contract tests.
- `docker-compose.yml`: MLflow server, one-shot trainer, API.
- `.github/workflows/ci.yml`: lint-ish compile check and tests.

## Notes for production
This is a runnable starter, not a production-hardened platform. For production, use managed/remote object storage for artifacts, a secured MLflow server, secrets management, authentication/authorization, TLS, pinned image digests, data validation, drift monitoring, and a real domain dataset. The included dataset is synthetic and intended only to verify the pipeline end-to-end.
