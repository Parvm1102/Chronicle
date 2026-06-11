# Main app (FastAPI + mounted Gradio apps) for Hugging Face Spaces (sdk: docker).
#
# The Gradio SDK can only launch a single Blocks app; this app instead mounts
# three Gradio apps onto a FastAPI server with custom /tts and /chat routes, so
# it runs its own uvicorn server. A Docker Space gives us full control over how
# the process starts (just `python app.py`), avoiding the port collision the
# Gradio SDK runner causes.
#
# The heavy GPU TTS service (tts_service/) is NOT part of this image — deploy it
# separately to Modal and point TTS_SERVICE_URL at it. PostgreSQL and Qdrant are
# expected to be managed services (Supabase / Qdrant Cloud) configured via the
# Space's secrets.

FROM python:3.12-slim

# libgomp1: required by onnxruntime (pulled in by fastembed for RAG embeddings).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (Hugging Face Spaces convention).
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/home/user/app/data/hf

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app.py"]
