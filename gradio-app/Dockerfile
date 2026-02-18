# =============================================================================
# Dockerfile para Gradio App + NLLB Translation
# Interfaz de usuario con traducción integrada
# =============================================================================

FROM python:3.11-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar modelo NLLB (reduce cold start significativamente)
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
print('Descargando NLLB-200-distilled-600M...'); \
AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); \
AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M'); \
print('✅ Modelo descargado')"

# Copiar código de la aplicación
COPY app.py .

# Puerto de Gradio
EXPOSE 7860

# Variables de entorno para conexión a vLLM
ENV VLLM_HOST=localhost
ENV VLLM_PORT=8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Comando de inicio
CMD ["python", "app.py"]
