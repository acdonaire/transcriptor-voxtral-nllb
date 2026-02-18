#!/bin/bash
# =============================================================================
# Script de inicio para Verda Cloud
# Ejecuta ambos contenedores manualmente (sin docker-compose)
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 Iniciando Voxtral + NLLB en Verda Cloud                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Configuración
VLLM_PORT=8000
GRADIO_PORT=7860
HF_CACHE_DIR="/root/.cache/huggingface"

# =============================================================================
# Paso 1: Iniciar servidor vLLM con Voxtral
# =============================================================================
echo ""
echo "📦 [1/2] Iniciando servidor vLLM con Voxtral-Mini-4B..."
echo "     Puerto: ${VLLM_PORT}"
echo ""

# Instalar dependencias de audio para vLLM
pip install -q soxr librosa soundfile mistral-common>=1.9.0

# Iniciar vLLM en background
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
    --host 0.0.0.0 \
    --port ${VLLM_PORT} \
    --compilation_config '{"cudagraph_mode": "PIECEWISE"}' \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.65 \
    &

VLLM_PID=$!
echo "     PID de vLLM: ${VLLM_PID}"

# Esperar a que vLLM esté listo
echo "     Esperando a que vLLM esté listo..."
MAX_WAIT=300  # 5 minutos máximo
WAITED=0
while ! curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; do
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "❌ Error: vLLM no respondió después de ${MAX_WAIT} segundos"
        exit 1
    fi
    echo "     Esperando... (${WAITED}s)"
done
echo "✅ vLLM está listo!"

# =============================================================================
# Paso 2: Iniciar aplicación Gradio
# =============================================================================
echo ""
echo "📦 [2/2] Iniciando aplicación Gradio con NLLB..."
echo "     Puerto: ${GRADIO_PORT}"
echo ""

# Instalar dependencias de Gradio
pip install -q gradio>=4.44.0 transformers torch accelerate sentencepiece \
    numpy soxr soundfile librosa websockets

# Configurar variables de entorno
export VLLM_HOST=localhost
export VLLM_PORT=${VLLM_PORT}

# Ejecutar la aplicación Gradio
python /app/app.py &

GRADIO_PID=$!
echo "     PID de Gradio: ${GRADIO_PID}"

# =============================================================================
# Mantener el script corriendo
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Servicios iniciados correctamente                        ║"
echo "║                                                              ║"
echo "║  🔗 Gradio UI:    http://localhost:${GRADIO_PORT}              ║"
echo "║  🔗 vLLM API:     http://localhost:${VLLM_PORT}               ║"
echo "║  🔗 Realtime WS:  ws://localhost:${VLLM_PORT}/v1/realtime     ║"
echo "║                                                              ║"
echo "║  Presiona Ctrl+C para detener todos los servicios            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Manejar señales de terminación
cleanup() {
    echo ""
    echo "🛑 Deteniendo servicios..."
    kill $VLLM_PID 2>/dev/null || true
    kill $GRADIO_PID 2>/dev/null || true
    echo "👋 Servicios detenidos"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Esperar a que los procesos terminen
wait
