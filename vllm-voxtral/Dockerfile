# =============================================================================
# Dockerfile para vLLM con Voxtral-Mini-4B-Realtime
# Servidor de transcripción en tiempo real via WebSocket
# =============================================================================

FROM vllm/vllm-openai:latest

# Instalar dependencias de audio requeridas por Voxtral
RUN pip install --no-cache-dir \
    soxr \
    librosa \
    soundfile \
    mistral-common>=1.9.0

# Variables de entorno
ENV VLLM_DISABLE_COMPILE_CACHE=1
ENV HF_HOME=/root/.cache/huggingface

# Pre-descargar el modelo (opcional, reduce cold start)
# Comentar si prefieres descarga en runtime
ARG HF_TOKEN
RUN --mount=type=secret,id=hf_token,target=/run/secrets/hf_token \
    if [ -f /run/secrets/hf_token ]; then \
        export HF_TOKEN=$(cat /run/secrets/hf_token); \
    fi && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('mistralai/Voxtral-Mini-4B-Realtime-2602')" || true

# Puerto para la API
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
# --compilation_config con cudagraph_mode PIECEWISE para mejor compatibilidad
CMD ["vllm", "serve", \
     "mistralai/Voxtral-Mini-4B-Realtime-2602", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--compilation_config", "{\"cudagraph_mode\": \"PIECEWISE\"}", \
     "--max-model-len", "32768"]
