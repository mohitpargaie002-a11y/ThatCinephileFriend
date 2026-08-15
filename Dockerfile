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

# Pre-download tokenizer and ONNX model at build time
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'tokenizer.json'); hf_hub_download('sentence-transformers/all-MiniLM-L6-v2', 'onnx/model_O4.onnx')"

COPY app ./app
COPY frontend ./frontend

RUN chown -R appuser:appuser /code /opt/huggingface
USER appuser

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
