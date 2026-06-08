# Chatterbox-Turbo TTS microservice (GPU).
#
# Build from the REPOSITORY ROOT so the build context can see both the
# chatterbox source and tts_service/:
#
#   docker build -f tts_service/Dockerfile -t novel-reader-tts .
#
# (docker-compose.yml already sets context: . and this dockerfile.)
#
# Run (needs the NVIDIA Container Toolkit for --gpus):
#   docker run --gpus all -p 8070:8070 \
#     -v "$PWD/voice_samples:/voice_samples:ro" \
#     -v "$PWD/data/tts_cache:/data/tts_cache" \
#     -v "$PWD/data/hf:/data/hf" \
#     novel-reader-tts

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# git: resemble-perth installs from a git URL. libsndfile1/ffmpeg: soundfile + librosa I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Chatterbox first (its pinned torch==2.6.0 matches the base image).
COPY chatterbox-trying/chatterbox /app/chatterbox
RUN pip install --no-cache-dir /app/chatterbox

# Thin web layer deps.
COPY tts_service/requirements.txt /app/tts_service/requirements.txt
RUN pip install --no-cache-dir -r /app/tts_service/requirements.txt

# Service code.
COPY tts_service /app/tts_service

ENV TTS_DEVICE=cuda \
    VOICE_SAMPLES_DIR=/voice_samples \
    TTS_CACHE_DIR=/data/tts_cache \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1

EXPOSE 8070

CMD ["uvicorn", "tts_service.server:app", "--host", "0.0.0.0", "--port", "8070"]
