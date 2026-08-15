FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /code

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m appuser && \
    mkdir -p /opt/huggingface && chown -R appuser:appuser /opt/huggingface /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download both PyTorch and ONNX O4 optimized models at build time for instant zero-cold-start startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', backend='onnx', model_kwargs={'file_name': 'onnx/model_O4.onnx'})"

COPY app ./app
COPY frontend ./frontend

RUN chown -R appuser:appuser /code /opt/huggingface
USER appuser

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
