# 🎤 Voxtral + NLLB: Transcripción y Traducción en Tiempo Real

Sistema de transcripción de voz en tiempo real con traducción automática, diseñado para workshops de IA.

## 📋 Descripción

Esta aplicación combina dos modelos de IA:

| Modelo | Función | Parámetros | VRAM |
|--------|---------|------------|------|
| **Voxtral-Mini-4B-Realtime** | Transcripción en tiempo real | 4.4B | ~16GB |
| **NLLB-200-distilled-600M** | Traducción a inglés | 600M | ~3GB |

### Características

- ✅ Transcripción en tiempo real con latencia <500ms
- ✅ Detección automática de idioma
- ✅ Soporte para 13 idiomas (transcripción)
- ✅ Traducción a inglés desde 200+ idiomas
- ✅ Interfaz web con Gradio
- ✅ API WebSocket compatible con vLLM Realtime

## 🔧 Requisitos de Hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU | A10 24GB | A100 40GB/80GB |
| RAM | 16GB | 32GB |
| Disco | 50GB | 100GB (para cache) |

## 🚀 Despliegue en Verda Cloud

### Opción 1: Contenedor Unificado (Recomendado)

1. **Crear instancia en Verda:**
   - GPU: A100 40GB o superior
   - Imagen base: `vllm/vllm-openai:latest`
   - Puertos expuestos: `7860`, `8000`

2. **Clonar repositorio:**
   ```bash
   git clone https://github.com/acdonaire/transcriptor-voxtral-nllb.git
   cd transcriptor-voxtral-nllb
   ```

3. **Construir y ejecutar:**
   ```bash
   docker build -t voxtral-nllb:latest .
   docker run --gpus all -p 7860:7860 -p 8000:8000 voxtral-nllb:latest
   ```

### Opción 2: Docker Compose (dos contenedores)

```bash
docker-compose up -d
```

### Opción 3: Ejecución directa en Verda

Si usas una instancia con vLLM preinstalado:

```bash
# Terminal 1: Iniciar vLLM
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
    --host 0.0.0.0 \
    --port 8000 \
    --compilation_config '{"cudagraph_mode": "PIECEWISE"}' \
    --max-model-len 32768

# Terminal 2: Iniciar Gradio (después de que vLLM esté listo)
pip install gradio transformers websockets soxr
python gradio-app/app.py
```

## 📁 Estructura del Proyecto

```
voxtral-nllb-verda/
├── Dockerfile              # Imagen unificada
├── docker-compose.yml      # Orquestación de 2 contenedores
├── supervisord.conf        # Gestión de procesos
├── start.sh               # Script de inicio manual
├── README.md              # Este archivo
├── vllm-voxtral/
│   └── Dockerfile         # Solo vLLM + Voxtral
└── gradio-app/
    ├── Dockerfile         # Solo Gradio + NLLB
    ├── app.py            # Aplicación principal
    └── requirements.txt   # Dependencias Python
```

## 🌐 Acceso a la Aplicación

Una vez desplegado:

- **Interfaz Gradio**: `http://<IP_VERDA>:7860`
- **API vLLM**: `http://<IP_VERDA>:8000`
- **WebSocket Realtime**: `ws://<IP_VERDA>:8000/v1/realtime`

## 📖 Uso

1. Abre la interfaz Gradio en tu navegador
2. Haz clic en **🎤 Iniciar**
3. Permite el acceso al micrófono
4. Habla en cualquier idioma soportado
5. La transcripción aparece en tiempo real
6. La traducción al inglés se genera automáticamente
7. Haz clic en **⏹️ Detener** cuando termines

## 🌍 Idiomas Soportados

### Transcripción (Voxtral)
Español, Inglés, Francés, Alemán, Italiano, Portugués, Holandés, Ruso, Chino, Japonés, Coreano, Árabe, Hindi

### Traducción (NLLB)
200+ idiomas incluyendo todos los anteriores y muchos más

## ⚙️ Configuración Avanzada

### Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `VLLM_HOST` | Host del servidor vLLM | `localhost` |
| `VLLM_PORT` | Puerto del servidor vLLM | `8000` |
| `GRADIO_PORT` | Puerto de la interfaz Gradio | `7860` |
| `HF_TOKEN` | Token de HuggingFace (opcional) | - |

### Ajustes de VRAM

Para GPUs con menos memoria, ajusta `--gpu-memory-utilization`:

```bash
# Para A10 24GB
--gpu-memory-utilization 0.85

# Para A100 40GB
--gpu-memory-utilization 0.70

# Para A100 80GB
--gpu-memory-utilization 0.50
```

### Ajuste de Latencia

El delay de transcripción se puede configurar (480ms es el sweet spot):

```bash
# En params.json del modelo
"transcription_delay_ms": 480  # 80ms a 2400ms
```

## 🐛 Solución de Problemas

### vLLM no inicia
```bash
# Verificar GPU disponible
nvidia-smi

# Probar con modo eager
vllm serve ... --enforce-eager
```

### WebSocket no conecta
```bash
# Verificar que vLLM esté corriendo
curl http://localhost:8000/health

# Ver logs
docker logs vllm-voxtral
```

### Error de VRAM
```bash
# Reducir utilización de memoria
--gpu-memory-utilization 0.5

# Reducir contexto máximo
--max-model-len 16384
```

## 📜 Licencias

- **Voxtral-Mini-4B**: Apache 2.0 ✅
- **NLLB-200**: CC-BY-NC-4.0 ⚠️ (solo uso no comercial)

## 🔗 Referencias

- [Voxtral Model Card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [NLLB-200 Model Card](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [vLLM Realtime API](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)
- [Verda Cloud](https://verda.com)

---

**ColoqIALab** - Workshop de IA | Febrero 2026
