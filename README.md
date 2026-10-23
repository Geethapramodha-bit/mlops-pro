# Agentic RAG MLOps

An AI-powered application combining a traditional customer-support chatbot with an Agentic AI learning assistant. The project uses machine learning, Retrieval-Augmented Generation (RAG), LangGraph, FastAPI, ChromaDB, Ollama, and Docker.

## Project Overview

This project provides two AI workflows:

1. **Support Chatbot:** Classifies user queries into intents, routes them to appropriate handlers, generates responses, and records conversations and feedback.
2. **Agentic AI Learning Assistant:** Retrieves relevant information from a knowledge base and uses an LLM to generate context-grounded answers. When sufficient relevant information is unavailable, it returns a fallback response.

The repository also includes model evaluation, RAG evaluation, experiment tracking, and data/model pipeline tooling.

## Architecture

### Support Chatbot

1. A user submits a query through the FastAPI `/chat` endpoint.
2. The intent classifier predicts the query category.
3. The application routes the query to the appropriate handler.
4. The response is returned to the user.
5. Conversation records and feedback can be stored through the application endpoints.

### Agentic AI Learning Assistant

1. The user submits a question.
2. A SentenceTransformer model converts the question into an embedding.
3. ChromaDB retrieves the top matching knowledge-base documents.
4. A LangGraph workflow checks retrieval results and decides whether to answer or use the fallback path.
5. Ollama runs the local LLM to generate a response using retrieved context.
6. The API returns the answer and available source information.

## Technology Stack

- **Language:** Python 3.12
- **API:** FastAPI and Uvicorn
- **Machine Learning:** scikit-learn
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB
- **Agent Workflow:** LangGraph
- **LLM Runtime:** Ollama with `llama3:latest`
- **Experiment Tracking:** MLflow
- **Pipeline and Data Versioning:** DVC
- **Containerization:** Docker and Docker Compose
- **Testing:** pytest

## Requirements

- Python 3.12
- Git
- Docker Desktop with the Docker engine running
- Ollama installed on the host machine for local LLM inference
- The `llama3:latest` model available in Ollama

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd agentic-rag-mlops
```

Replace `<your-repository-url>` with your actual repository URL.

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Use the project's main dependency file:

```powershell
pip install -r requirements.txt
```

If your environment uses a different dependency-management workflow, follow the dependency instructions in the repository.

### 4. Prepare the local LLM

Start Ollama and ensure the model is available:

```powershell
ollama pull llama3:latest
```

### 5. Start the API

From the repository root:

```powershell
uvicorn app.main:app --reload
```

The API is available at:

- API root: `http://localhost:8000/`
- Health check: `http://localhost:8000/health`
- Interactive API documentation: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application root |
| GET | `/health` | Health check |
| POST | `/chat` | Support chatbot |
| POST | `/feedback` | Record feedback |
| POST | `/agentic-ai` | Agentic AI knowledge assistant |

Use `/docs` to inspect the request schemas and try the endpoints interactively.

## Running Tests

Run the automated tests from the repository root:

```powershell
python -m pytest -q
```

The latest recorded test run completed with **2 passed and 0 failed**, with one deprecation warning from the Starlette test client and AnyIO.

## RAG Evaluation

Run the retrieval evaluation as a Python module:

```powershell
python -m scripts.evaluate_agentic_ai_retrieval
```

The evaluation checks whether expected knowledge-base sources appear among the top three retrieved documents.

An additional answer evaluation can be run with:

```powershell
python -m scripts.evaluate_agentic_ai_answers
```

In the manual retrieval-distance check, three relevant questions returned best distances between approximately `0.60` and `0.66`. An unrelated question returned a distance of approximately `1.68`. These observations support testing the configured distance threshold of `1.0`, but the sample is too small to establish a generally optimal threshold.

## Docker Deployment

The project includes a `Dockerfile`, `compose.yaml`, and `requirements-docker.txt`.

1. Ensure Docker Desktop is running.
2. Confirm `.env.docker` exists and contains the required local runtime configuration.
3. Confirm the Ollama model is available on the host.
4. Build and start the service:

```powershell
docker compose up --build -d
```

5. Check the container status:

```powershell
docker compose ps
```

6. Test the health endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

7. Stop the service when finished:

```powershell
docker compose down
```

**Deployment note:** The current Compose configuration uses a Windows-specific Hugging Face cache path and connects to Ollama on the host. Adjust these settings when deploying on another machine or operating system. The Docker image contains the application, model files, and data copied by the Dockerfile; the Compose file also mounts local data and the embedding-model cache.

## MLOps Components

The repository includes tooling for:

- Training and evaluating a support-intent classifier.
- Logging model metrics and experiments with MLflow.
- Managing pipeline stages and data artifacts with DVC.
- Building the Agentic AI retrieval knowledge base.
- Evaluating retrieval behavior and generated answers.

Check the individual scripts and `dvc.yaml` for the pipeline stages and the commands supported by the current repository.

## Limitations and Future Improvements

- Expand automated test coverage beyond the current two tests.
- Evaluate retrieval precision, recall, and fallback behavior over a larger and more diverse question set.
- Tune the ChromaDB distance threshold using a representative evaluation dataset.
- Improve response formatting and add stronger handling for LLM service failures.
- Make Docker cache paths configurable for other operating systems.
- Separate development and production configuration.
- Add CI automation for tests and deployment checks.
- Document reproducible dataset preparation and model-building steps.

## Security Notes

- Do not commit API keys, passwords, or other secrets.
- Keep `.env.docker` and local environment files out of version control.
- Do not commit conversation backups or sensitive user data.
- Review generated reports and data files before publishing the repository.

## Project Status

The application has been tested locally and in Docker. The current checks cover basic API behavior, retrieval and fallback examples, and the existing automated tests. Broader testing and deployment hardening remain necessary before treating the project as production-ready.
