# DocuMind backend — FastAPI + LangGraph pipeline.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so code changes don't bust the layer cache.
# torch is pinned to the CPU-only wheel BEFORE requirements.txt so
# sentence-transformers doesn't pull the multi-GB CUDA build into the image.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
