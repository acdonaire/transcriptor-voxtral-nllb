# Transcriptor Voxtral + NLLB

Sistema de transcripción en tiempo real con traducción automática.

## Modelos

- **Transcripción**: Voxtral-Mini-4B-Realtime (Mistral AI) - 4.4B parámetros, ~16GB VRAM
- **Traducción**: NLLB-200-distilled-600M (Meta) - 600M parámetros, ~3GB VRAM

## Servicios

El sistema se compone de 3 contenedores independientes:

- **vllm-voxtral** (puerto 8000): Servidor vLLM con Voxtral para transcripción en tiempo real via WebSocket
- **nllb-api** (puerto 8001): API REST FastAPI para traducción con NLLB-200
- **gradio-app** (puerto 7860): Interfaz web con micrófono

## Idiomas soportados

**Transcripción (Voxtral):** Español, Inglés, Francés, Alemán, Italiano, Portugués, Holandés, Ruso, Chino, Japonés, Coreano, Árabe, Hindi

**Traducción (NLLB):** 200+ idiomas. Configurados en la app: Inglés, Español, Francés, Alemán, Italiano, Portugués, Holandés, Ruso, Chino, Japonés, Coreano, Árabe

## Requisitos

- GPU con mínimo 24GB VRAM (recomendado: L40S 48GB)
- Docker con soporte NVIDIA
- ~20GB espacio en disco

## Instalación
```bash
git clone https://github.com/acdonaire/transcriptor-voxtral-nllb.git
cd transcriptor-voxtral-nllb
```

## Uso

Iniciar servicios en orden:
```bash
# 1. Voxtral (esperar a "Application startup complete")
docker compose up -d vllm-voxtral
docker logs -f vllm-voxtral

# 2. NLLB (esperar a "NLLB cargado en cuda")
docker compose up -d nllb-api
docker logs -f nllb-api

# 3. Gradio
docker compose up -d gradio-app
docker logs -f gradio-app
```

Acceder a la interfaz en `http://<IP>:7860`

## Tests individuales

Para probar cada servicio por separado:
```bash
# Instalar dependencias
pip3 install gradio websockets numpy httpx --break-system-packages

# Test Voxtral (transcripción)
python3 test_realtime.py

# Test NLLB (traducción)
python3 test_nllb.py
```

## API NLLB
```bash
curl -X POST http://localhost:8001/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hola mundo","source_lang":"es","target_lang":"en"}'
```

Respuesta:
```json
{"translation":"Hello world","source_lang":"es","target_lang":"en"}
```

## Características

- Transcripción en tiempo real via WebSocket
- Detección automática de idioma
- Traducción con buffer inteligente (debounce 1.5s + mínimo 10 caracteres nuevos)
- Selector de idioma destino
- Limpieza automática al reiniciar grabación

## Estructura del proyecto
```
transcriptor-voxtral-nllb/
├── docker-compose.yml
├── Dockerfile.vllm
├── gradio-app/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
├── nllb-api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── server.py
├── test_realtime.py
├── test_nllb.py
└── README.md
```

## Códigos de idioma NLLB

El servidor NLLB convierte códigos ISO 639-1 a códigos FLORES-200:

| Código | NLLB |
|--------|------|
| es | spa_Latn |
| en | eng_Latn |
| fr | fra_Latn |
| de | deu_Latn |
| it | ita_Latn |
| pt | por_Latn |
| nl | nld_Latn |
| ru | rus_Cyrl |
| zh | zho_Hans |
| ja | jpn_Jpan |
| ko | kor_Hang |
| ar | arb_Arab |

## Licencia

MIT
