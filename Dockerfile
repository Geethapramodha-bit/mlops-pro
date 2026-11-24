FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
RUN pip install --no-cache-dir email-validator==2.3.0

# Download and cache the embedding model during image build.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy backend application.
COPY app/ ./app/

# Copy only the raw datasets required for RAG initialization.
COPY data/raw/agentic_ai_dataset.json ./data/raw/
COPY data/raw/knowledge_base.txt ./data/raw/
COPY data/raw/support_intents.csv ./data/raw/
COPY data/raw/agentic_ai/ ./data/raw/agentic_ai/

# Copy training and RAG builder scripts.
COPY scripts/train_support_model.py ./scripts/
COPY scripts/build_rag.py ./scripts/
COPY scripts/build_agentic_ai_rag.py ./scripts/
COPY scripts/initialize_render_data.py ./scripts/

# Prepare the classifier and both ChromaDB collections.
RUN python scripts/train_support_model.py
RUN python scripts/build_rag.py
RUN python scripts/build_agentic_ai_rag.py

EXPOSE 8000

CMD ["sh", "-c", "python scripts/initialize_render_data.py && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]


