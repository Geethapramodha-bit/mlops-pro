# Agentic RAG MLOps Chatbot

An AI-powered customer-support chatbot and Agentic AI learning assistant, supported by an end-to-end MLOps workflow for model training, evaluation, experiment tracking, data versioning, deployment and monitoring.

## 1. Project Overview

This project combines two AI workflows:

**1. Customer Support Chatbot**
- Classifies user queries into support intents.
- Routes queries to appropriate handlers.
- Generates responses based on predicted intent.
- Records conversations and user feedback.

**2. Agentic AI Learning Assistant**
- Retrieves relevant information from a knowledge base.
- Uses ChromaDB for vector similarity search.
- Uses LangGraph to manage the agent workflow.
- Uses an LLM through Ollama to generate context-grounded responses.
- Provides fallback responses when relevant knowledge is unavailable.

The project integrates MLOps practices to make model development, evaluation, tracking, versioning and monitoring reproducible.

## 2. Architecture

### Application workflow

1. User submits a query through the chatbot interface or FastAPI.
2. The NLP intent classifier predicts the query category.
3. The application routes the request to the appropriate handler.
4. The Agentic AI workflow retrieves knowledge when required.
5. The LLM generates a response using available context.
6. The application returns the response and records conversation data.

### MLOps workflow

**Data Versioning → Model Training → Evaluation → Experiment Tracking → Model Registry → Deployment → Monitoring**

| Component | Responsibility |
|---|---|
| NLP | Conversational intent classification |
| FastAPI | Application API |
| MLflow | Experiment tracking, metrics and model registry |
| DVC | Data and pipeline versioning |
| Docker | Application containerization |
| GitHub Actions | Automated testing and CI |
| Evidently AI | Data drift monitoring |

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.12 |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF, Logistic Regression |
| API | FastAPI, Uvicorn |
| Agent Framework | LangGraph |
| LLM Runtime | Ollama |
| Embeddings | SentenceTransformers |
| Vector Database | ChromaDB |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Monitoring | Evidently AI |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Testing | Pytest |
| Frontend | React |

## 4. Project Structure

```text
agentic-rag-mlops/
│
├── app/
│   ├── agents/
│   ├── api/
│   ├── mlops/
│   │   └── evaluation_gate.py
│   ├── rag/
│   ├── services/
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── conversations/
│
├── models/
│
├── frontend/
│
├── scripts/
│   ├── train_model.py
│   ├── evaluate_evidently.py
│   └── ...
│
├── reports/
│
├── tests/
│   └── test_evaluation_gate.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── dvc.yaml
├── dvc.lock
├── Dockerfile
├── compose.yaml
├── requirements.txt
└── README.md
```

## 5. Requirements

Install the following:

- Python 3.12
- Git
- Docker Desktop
- Ollama
- Required Ollama model (`llama3:latest` for the documented configuration)

## 6. Local Setup

### Clone the repository

```powershell
git clone https://github.com/Geethapramodha-bit/mlops-pro.git
cd mlops-pro
```

### Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Configure the session secret

Create a local `.env` file containing a secure `BOTPLAYER_SESSION_SECRET`.

Example generation command:

```powershell
python -c "import secrets; from pathlib import Path; Path('.env').write_text('BOTPLAYER_SESSION_SECRET=' + secrets.token_urlsafe(32) + '\n', encoding='utf-8')"
```

Do not commit the generated secret.

### Prepare Ollama

Start Ollama and make sure the required model is available:

```powershell
ollama pull llama3:latest
ollama list
```

## 7. Run the Application

From the project root:

```powershell
uvicorn app.main:app --reload
```

Application URLs:

| Resource | URL |
|---|---|
| API root | http://localhost:8000/ |
| Health check | http://localhost:8000/health |
| Swagger documentation | http://localhost:8000/docs |
| OpenAPI schema | http://localhost:8000/openapi.json |

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Application information |
| GET | `/health` | API health check |
| POST | `/chat` | Customer support chatbot |
| POST | `/feedback` | Record user feedback |
| POST | `/agentic-ai` | Agentic AI learning assistant |

Use Swagger UI at `/docs` to inspect request schemas and test the endpoints.

## 9. Model Training and Evaluation

The intent-classification pipeline uses:

- TF-IDF for text feature extraction.
- Logistic Regression for intent classification.
- Train, validation and test datasets for evaluation.
- Accuracy, precision, recall and F1-score for performance measurement.

Run model training from the repository root:

```powershell
python -m scripts.train_model
```

### Evaluation Gate

The evaluation gate checks whether the trained model meets minimum validation performance requirements.

Current thresholds:

- Validation accuracy: 0.85
- Validation F1-score: 0.85

The model must satisfy these conditions to pass the evaluation gate.

### Recorded Model Results

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.8887 | 0.8938 |
| Weighted Precision | 0.8970 | 0.8991 |
| Weighted Recall | 0.8887 | 0.8938 |
| Weighted F1-score | 0.8882 | 0.8928 |

These are the recorded results from the current model training run.

## 10. MLflow Experiment Tracking and Model Registry

MLflow is used to track model experiments, parameters, metrics and model artifacts.

The project uses an SQLite tracking backend:

```text
sqlite:///mlflow.db
```

Tracked metrics include:

- Accuracy
- Precision
- Recall
- F1-score

A trained intent-classification model has also been registered in the MLflow Model Registry.

Current registered model:

```text
Model: intent_classifier
Version: 1
Status: READY
```

### Open the MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

http://localhost:5000

## 11. DVC Data and Pipeline Versioning

DVC is used to track datasets and make pipeline execution reproducible.

The training pipeline is defined in:

```text
dvc.yaml
```

Check pipeline status:

```powershell
dvc status
```

Reproduce the pipeline:

```powershell
dvc repro
```

DVC tracks the relevant dependencies, pipeline stages and model artifacts.

A DVC remote must be configured separately before pushing versioned data to remote storage.

## 12. Evidently AI Monitoring

Evidently AI is used to monitor changes in conversation data.

The monitoring workflow compares reference and current conversation datasets to identify potential data drift.

Generate the report:

```powershell
python -m scripts.evaluate_evidently
```

Generated report:

```text
reports/evidently_report.html
```

Open the HTML report in a browser to inspect the monitoring results.

**Note:** The current conversation dataset is relatively small. Drift findings should be interpreted as preliminary rather than statistically conclusive.

## 13. Docker Containerization

The project includes:

- `Dockerfile`
- `compose.yaml`
- `requirements-docker.txt`
- `.env.docker`

Ensure Docker Desktop is running and the local runtime configuration is available.

Build and start the API:

```powershell
docker compose up --build -d
```

Check containers:

```powershell
docker compose ps
```

Test the API:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Stop the container:

```powershell
docker compose down
```

**Configuration note:** The current Compose configuration uses a host-specific Hugging Face cache mount and connects to Ollama through the host machine. Update these settings when running on another computer or operating system.

## 14. GitHub Actions CI

GitHub Actions provides continuous integration for the project.

Workflow file:

```text
.github/workflows/ci.yml
```

The workflow:

1. Checks out the repository.
2. Sets up Python 3.12.
3. Installs CI dependencies.
4. Runs automated tests using Pytest.

It runs on configured pushes, pull requests and manual workflow dispatches.

**Current CI status:** The workflow associated with commit `cbce8ed` on the `master` branch completed successfully.

The current workflow focuses on testing. Automated model retraining and automated deployment are not part of the implemented workflow.

## 15. Automated Testing

Run the test suite:

```powershell
python -m pytest -q
```

Latest recorded local result:

```text
6 passed, 1 warning
```

The warning was a Starlette/AnyIO deprecation warning and did not cause test failure.

## 16. Current Implementation Status

| Component | Status |
|---|---|
| NLP intent classifier | Implemented |
| Agentic RAG workflow | Implemented |
| FastAPI endpoints | Tested |
| MLflow experiment tracking | Implemented |
| MLflow model registration | Verified |
| DVC pipeline | Executed successfully |
| Evaluation gate | Implemented and tested |
| Docker configuration | Validated |
| GitHub Actions CI | Passed for latest commit |
| Evidently AI report | Generated |
| Automated tests | 6 passed |

## 17. Limitations and Future Improvements

- Increase the size and diversity of conversation data for monitoring.
- Configure remote DVC storage for shared dataset versioning.
- Expand end-to-end testing across different chatbot intents and failure scenarios.
- Improve retrieval quality through a larger evaluation dataset.
- Make Docker configuration portable across operating systems.
- Add automated model deployment and retraining only when required.
- Improve production monitoring and model lifecycle management.

## 18. Security Notes

- Never commit API keys, passwords or session secrets.
- Keep `.env` and other local environment files private.
- Avoid committing conversation backups or sensitive user information.
- Review generated reports and datasets before publishing.
- Use secure configuration for production deployment.

## 19. Repository

GitHub:

https://github.com/Geethapramodha-bit/mlops-pro

Branch: `master`

Latest documented commit:

```text
cbce8ed
```

---

**Project:** Agentic RAG MLOps Chatbot

**Focus:** NLP, Agentic AI, Model Lifecycle Management, Experiment Tracking, Data Versioning, CI and Monitoring.
