#!/usr/bin/env python3
"""
=============================================================================
Aplicación de Transcripción en Tiempo Real + Traducción
- Voxtral-Mini-4B para transcripción (via vLLM WebSocket)
- NLLB-200-distilled-600M para traducción a inglés
=============================================================================
"""

import os
import asyncio
import base64
import json
import queue
import threading
from typing import Optional

import gradio as gr
import numpy as np
import websockets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# =============================================================================
# Configuración
# =============================================================================

VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
VOXTRAL_MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

SAMPLE_RATE = 16_000  # Voxtral requiere 16kHz

# Mapeo de códigos de idioma detectados a códigos NLLB
LANG_TO_NLLB = {
    "es": "spa_Latn",  # Español
    "en": "eng_Latn",  # Inglés
    "fr": "fra_Latn",  # Francés
    "de": "deu_Latn",  # Alemán
    "it": "ita_Latn",  # Italiano
    "pt": "por_Latn",  # Portugués
    "nl": "nld_Latn",  # Holandés
    "ru": "rus_Cyrl",  # Ruso
    "zh": "zho_Hans",  # Chino simplificado
    "ja": "jpn_Jpan",  # Japonés
    "ko": "kor_Hang",  # Coreano
    "ar": "arb_Arab",  # Árabe
    "hi": "hin_Deva",  # Hindi
}

# =============================================================================
# Cargar modelo NLLB para traducción
# =============================================================================

print(f"🔄 Cargando modelo de traducción: {NLLB_MODEL}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Dispositivo: {device}")

nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
    NLLB_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

print("✅ Modelo NLLB cargado correctamente")

# =============================================================================
# Estado global para WebSocket
# =============================================================================

audio_queue: queue.Queue = queue.Queue()
transcription_text = ""
detected_language = "es"  # Por defecto español
is_running = False
ws_thread: Optional[threading.Thread] = None


def translate_text(text: str, source_lang: str = "es", target_lang: str = "en") -> str:
    """Traduce texto usando NLLB-200."""
    if not text.strip():
        return ""
    
    # Si el idioma fuente es igual al destino, no traducir
    if source_lang == target_lang:
        return text
    
    # Obtener códigos NLLB
    src_code = LANG_TO_NLLB.get(source_lang, "spa_Latn")
    tgt_code = LANG_TO_NLLB.get(target_lang, "eng_Latn")
    
    # Configurar idioma fuente en el tokenizer
    nllb_tokenizer.src_lang = src_code
    
    # Tokenizar
    inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Obtener el token ID del idioma destino
    forced_bos_token_id = nllb_tokenizer.convert_tokens_to_ids(tgt_code)
    
    # Generar traducción
    with torch.no_grad():
        outputs = nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True
        )
    
    # Decodificar
    translation = nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation


async def websocket_handler(update_callback):
    """Conecta al WebSocket de vLLM y maneja streaming de audio + transcripción."""
    global transcription_text, is_running, detected_language
    
    ws_url = f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime"
    print(f"🔌 Conectando a {ws_url}")
    
    try:
        async with websockets.connect(ws_url) as ws:
            # Esperar session.created
            response = json.loads(await ws.recv())
            if response.get("type") == "session.created":
                print(f"✅ Sesión creada: {response.get('id')}")
            
            # Configurar modelo
            await ws.send(json.dumps({
                "type": "session.update",
                "model": VOXTRAL_MODEL
            }))
            
            # Señalar que estamos listos
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            
            async def send_audio():
                """Envía chunks de audio al servidor."""
                while is_running:
                    try:
                        chunk = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: audio_queue.get(timeout=0.1)
                        )
                        await ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": chunk
                        }))
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Error enviando audio: {e}")
                        break
            
            async def receive_transcription():
                """Recibe transcripciones del servidor."""
                global transcription_text, detected_language
                while is_running:
                    try:
                        response = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                        
                        if response.get("type") == "transcription.delta":
                            delta = response.get("delta", "")
                            transcription_text += delta
                            
                            # Detectar idioma si está disponible
                            if "language" in response:
                                detected_language = response["language"]
                            
                            # Actualizar UI
                            update_callback(transcription_text, detected_language)
                            
                        elif response.get("type") == "transcription.done":
                            final_text = response.get("text", transcription_text)
                            transcription_text = final_text
                            update_callback(transcription_text, detected_language)
                            
                        elif response.get("type") == "error":
                            print(f"❌ Error: {response.get('error')}")
                            
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if is_running:
                            print(f"Error recibiendo: {e}")
                        break
            
            # Ejecutar envío y recepción en paralelo
            await asyncio.gather(send_audio(), receive_transcription())
            
    except Exception as e:
        print(f"❌ Error de conexión WebSocket: {e}")


def run_websocket_loop(update_callback):
    """Ejecuta el event loop de asyncio en un thread separado."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_handler(update_callback))


# =============================================================================
# Interfaz Gradio
# =============================================================================

def process_audio(audio_data, state):
    """Procesa audio del micrófono y lo envía al WebSocket."""
    global is_running
    
    if audio_data is None:
        return state.get("transcription", ""), state.get("translation", ""), state
    
    sample_rate, audio_array = audio_data
    
    # Convertir a mono si es estéreo
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    
    # Resamplear a 16kHz si es necesario
    if sample_rate != SAMPLE_RATE:
        import soxr
        audio_array = soxr.resample(audio_array.astype(np.float32), sample_rate, SAMPLE_RATE)
    
    # Normalizar a int16
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        audio_array = (audio_array * 32767).astype(np.int16)
    elif audio_array.dtype != np.int16:
        audio_array = audio_array.astype(np.int16)
    
    # Convertir a base64 PCM16
    audio_bytes = audio_array.tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    # Poner en la cola para el WebSocket
    if is_running:
        audio_queue.put(audio_base64)
    
    return state.get("transcription", ""), state.get("translation", ""), state


def start_recording():
    """Inicia la grabación y conexión WebSocket."""
    global is_running, ws_thread, transcription_text
    
    if is_running:
        return "⏹️ Ya está grabando...", "", {}
    
    is_running = True
    transcription_text = ""
    
    state = {"transcription": "", "translation": "", "language": "es"}
    
    def update_callback(text, lang):
        state["transcription"] = text
        state["language"] = lang
        # Traducir en tiempo real
        if text.strip():
            state["translation"] = translate_text(text, lang, "en")
    
    # Iniciar WebSocket en thread separado
    ws_thread = threading.Thread(target=run_websocket_loop, args=(update_callback,), daemon=True)
    ws_thread.start()
    
    return "🎤 Grabando... (habla ahora)", "", state


def stop_recording(state):
    """Detiene la grabación."""
    global is_running
    is_running = False
    
    # Limpiar cola de audio
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except:
            break
    
    # Traducción final
    final_transcription = state.get("transcription", "")
    final_lang = state.get("language", "es")
    final_translation = translate_text(final_transcription, final_lang, "en") if final_transcription else ""
    
    return f"✅ Detenido. Idioma detectado: {final_lang}", final_transcription, final_translation, state


def update_display(state):
    """Actualiza los campos de texto con el estado actual."""
    return (
        state.get("transcription", ""),
        state.get("translation", ""),
        f"Idioma: {state.get('language', 'es')}"
    )


# =============================================================================
# Crear interfaz
# =============================================================================

with gr.Blocks(
    title="🎤 Transcripción + Traducción en Tiempo Real",
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown("""
    # 🎤 Transcripción y Traducción en Tiempo Real
    
    **Modelos utilizados:**
    - **Transcripción**: Voxtral-Mini-4B-Realtime (Mistral AI) - 13 idiomas
    - **Traducción**: NLLB-200-distilled-600M (Meta) - 200+ idiomas
    
    ---
    """)
    
    state = gr.State({"transcription": "", "translation": "", "language": "es"})
    
    with gr.Row():
        with gr.Column(scale=1):
            status = gr.Textbox(
                label="Estado",
                value="⏸️ Listo para grabar",
                interactive=False
            )
            
            with gr.Row():
                start_btn = gr.Button("🎤 Iniciar", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Detener", variant="stop", size="lg")
            
            audio_input = gr.Audio(
                sources=["microphone"],
                streaming=True,
                label="Micrófono",
                type="numpy"
            )
            
            lang_display = gr.Textbox(
                label="Idioma Detectado",
                value="Español (es)",
                interactive=False
            )
    
    with gr.Row():
        with gr.Column():
            transcription_output = gr.Textbox(
                label="📝 Transcripción (idioma original)",
                placeholder="La transcripción aparecerá aquí...",
                lines=8,
                max_lines=15,
                interactive=False
            )
        
        with gr.Column():
            translation_output = gr.Textbox(
                label="🌍 Traducción (Inglés)",
                placeholder="La traducción aparecerá aquí...",
                lines=8,
                max_lines=15,
                interactive=False
            )
    
    # Eventos
    start_btn.click(
        fn=start_recording,
        inputs=[],
        outputs=[status, transcription_output, state]
    )
    
    stop_btn.click(
        fn=stop_recording,
        inputs=[state],
        outputs=[status, transcription_output, translation_output, state]
    )
    
    audio_input.stream(
        fn=process_audio,
        inputs=[audio_input, state],
        outputs=[transcription_output, translation_output, state]
    )
    
    # Actualización periódica del display
    demo.load(
        fn=lambda s: (s.get("transcription", ""), s.get("translation", ""), f"Idioma: {s.get('language', 'es')}"),
        inputs=[state],
        outputs=[transcription_output, translation_output, lang_display],
        every=0.5
    )
    
    gr.Markdown("""
    ---
    ### 📋 Instrucciones:
    1. Haz clic en **🎤 Iniciar** para comenzar
    2. Permite el acceso al micrófono cuando el navegador lo solicite
    3. Habla en cualquier idioma soportado (español, inglés, francés, etc.)
    4. La transcripción aparecerá en tiempo real
    5. La traducción al inglés se generará automáticamente
    6. Haz clic en **⏹️ Detener** cuando termines
    
    ### 🌐 Idiomas soportados para transcripción:
    Español, Inglés, Francés, Alemán, Italiano, Portugués, Holandés, Ruso, Chino, Japonés, Coreano, Árabe, Hindi
    """)


if __name__ == "__main__":
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🎤 Transcripción + Traducción en Tiempo Real                ║
    ║                                                              ║
    ║  Voxtral-Mini-4B → Transcripción                            ║
    ║  NLLB-200-600M   → Traducción                               ║
    ║                                                              ║
    ║  vLLM Server: ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
