#!/bin/bash
# =============================================================================
# Script de Despliegue Rápido para Verda Cloud
# Ejecutar directamente en la instancia de Verda
# =============================================================================
#
# Uso:
#   curl -fsSL https://raw.githubusercontent.com/acdonaire/transcriptor-voxtral-nllb/main/deploy-verda.sh | bash
#
# O manualmente:
#   git clone https://github.com/acdonaire/transcriptor-voxtral-nllb.git
#   cd transcriptor-voxtral-nllb
#   ./deploy-verda.sh
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  🚀 Desplegando Voxtral + NLLB en Verda Cloud                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar si estamos en el directorio correcto
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${YELLOW}📥 Clonando repositorio...${NC}"
    git clone https://github.com/acdonaire/transcriptor-voxtral-nllb.git
    cd transcriptor-voxtral-nllb
fi

# Verificar GPU
echo -e "${YELLOW}🔍 Verificando GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi no encontrado. ¿Tienes GPU NVIDIA?${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo -e "${GREEN}✅ GPU detectada: ${GPU_INFO}${NC}"

# Verificar memoria GPU
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -lt 20000 ]; then
    echo -e "${YELLOW}⚠️  Advertencia: GPU con menos de 20GB VRAM. Puede haber problemas.${NC}"
fi

# Verificar Docker
echo -e "${YELLOW}🔍 Verificando Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker no encontrado${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker disponible${NC}"

# Verificar nvidia-docker
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo -e "${YELLOW}⚠️  nvidia-docker runtime no detectado. Intentando con --gpus all${NC}"
fi

# Opción de despliegue
echo ""
echo "Selecciona el método de despliegue:"
echo "  1) Docker Compose (recomendado - 2 contenedores separados)"
echo "  2) Contenedor único (más simple)"
echo "  3) Ejecución directa (sin Docker)"
echo ""
read -p "Opción [1]: " DEPLOY_OPTION
DEPLOY_OPTION=${DEPLOY_OPTION:-1}

case $DEPLOY_OPTION in
    1)
        echo ""
        echo -e "${YELLOW}📦 Iniciando con Docker Compose...${NC}"
        docker-compose up -d
        echo ""
        echo -e "${GREEN}✅ Contenedores iniciados${NC}"
        echo ""
        echo "Ver logs:"
        echo "  docker-compose logs -f"
        ;;
    2)
        echo ""
        echo -e "${YELLOW}📦 Construyendo imagen única...${NC}"
        docker build -t voxtral-nllb:latest .
        echo ""
        echo -e "${YELLOW}🚀 Iniciando contenedor...${NC}"
        docker run -d \
            --name voxtral-nllb \
            --gpus all \
            -p 7860:7860 \
            -p 8000:8000 \
            -v huggingface_cache:/root/.cache/huggingface \
            voxtral-nllb:latest
        echo ""
        echo -e "${GREEN}✅ Contenedor iniciado${NC}"
        echo ""
        echo "Ver logs:"
        echo "  docker logs -f voxtral-nllb"
        ;;
    3)
        echo ""
        echo -e "${YELLOW}📦 Instalando dependencias...${NC}"
        pip install -q vllm gradio transformers websockets soxr librosa soundfile mistral-common
        echo ""
        echo -e "${YELLOW}🚀 Iniciando servicios...${NC}"
        chmod +x start.sh
        ./start.sh
        ;;
    *)
        echo -e "${RED}Opción no válida${NC}"
        exit 1
        ;;
esac

# Mostrar información de acceso
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Despliegue completado                                    ║"
echo "║                                                              ║"
echo "║  🌐 Interfaz Gradio: http://$(hostname -I | awk '{print $1}'):7860     ║"
echo "║  🔌 API vLLM:        http://$(hostname -I | awk '{print $1}'):8000     ║"
echo "║                                                              ║"
echo "║  ⏳ Primera carga: ~5-10 min (descarga de modelos)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
