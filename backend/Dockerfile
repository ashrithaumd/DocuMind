# DocuMind backend — FastAPI + LangGraph pipeline.
# Same Python as the backend/.venv that produced the Phase 0 baseline.
FROM python:3.12.8-slim

WORKDIR /app

# Install dependencies first so code changes don't bust the layer cache.
# torch is installed from the CPU-only wheel BEFORE requirements.txt so
# sentence-transformers doesn't pull the multi-GB CUDA build into the image.
# Version matches backend/.venv and CI.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
