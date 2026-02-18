# =============================================================================
# Dockerfile Unificado para Verda Cloud
# vLLM + Voxtral + NLLB + Gradio en un solo contenedor
# =============================================================================
# 
# Este Dockerfile crea una imagen que contiene:
# - vLLM con soporte para Voxtral (transcripción en tiempo real)
# - NLLB-200 para traducción
# - Interfaz Gradio
#
# Uso en Verda:
#   1. Subir este Dockerfile y los archivos a GitHub
#   2. Crear instancia con GPU A100 40/80GB
#   3. Exponer puertos 7860 (Gradio) y 8000 (vLLM API)
# =============================================================================

FROM vllm/vllm-openai:latest

# Metadatos
LABEL maintainer="ColoqIALab"
LABEL description="Voxtral Transcription + NLLB Translation"
LABEL version="1.0"

# Variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV VLLM_HOST=localhost
ENV VLLM_PORT=8000
ENV GRADIO_PORT=7860

# Instalar dependencias adicionales del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
RUN pip install --no-cache-dir \
    gradio>=4.44.0 \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    sentencepiece>=0.2.0 \
    soxr>=0.3.7 \
    soundfile>=0.12.1 \
    librosa>=0.10.0 \
    websockets>=12.0 \
    mistral-common>=1.9.0

# Directorio de trabajo
WORKDIR /app

# Argumento para saltar descarga de modelos (útil en CI)
ARG SKIP_MODEL_DOWNLOAD=false

# Pre-descargar modelos (reduce cold start significativamente)
# NLLB-200 (~2.4GB)
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
    python -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
print('📥 Descargando NLLB-200-distilled-600M...'); \
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); \
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M'); \
print('✅ NLLB descargado')"; \
    fi

# Voxtral (~9GB) - comentar si prefieres descarga en runtime
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
    python -c "\
from huggingface_hub import snapshot_download; \
print('📥 Descargando Voxtral-Mini-4B-Realtime...'); \
snapshot_download('mistralai/Voxtral-Mini-4B-Realtime-2602'); \
print('✅ Voxtral descargado')"; \
    fi

# Copiar aplicación
COPY gradio-app/app.py /app/app.py

# Copiar script de inicio
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Configuración de Supervisor para manejar múltiples procesos
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Exponer puertos
EXPOSE 7860 8000

# Healthcheck
HEALTHCHECK --interval=60s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:7860/ && curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
