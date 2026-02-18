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
    "es": "spa_Latn",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
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
detected_language = "es"
is_running = False
ws_thread: Optional[threading.Thread] = None


def translate_text(text: str, source_lang: str = "es", target_lang: str = "en") -> str:
    if not text.strip():
        return ""
    if source_lang == target_lang:
        return text
    src_code = LANG_TO_NLLB.get(source_lang, "spa_Latn")
    tgt_code = LANG_TO_NLLB.get(target_lang, "eng_Latn")
    nllb_tokenizer.src_lang = src_code
    inputs = nllb_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    forced_bos_token_id = nllb_tokenizer.convert_tokens_to_ids(tgt_code)
    with torch.no_grad():
        outputs = nllb_model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True
        )
    translation = nllb_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation


async def websocket_handler(update_callback):
    global transcription_text, is_running, detected_language
    ws_url = f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime"
    print(f"🔌 Conectando a {ws_url}")
    try:
        async with websockets.connect(ws_url) as ws:
            response = json.loads(await ws.recv())
            if response.get("type") == "session.created":
                print(f"✅ Sesión creada: {response.get('id')}")
            await ws.send(json.dumps({"type": "session.update", "model": VOXTRAL_MODEL}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            
            async def send_audio():
                while is_running:
                    try:
                        chunk = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: audio_queue.get(timeout=0.1)
                        )
                        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Error enviando audio: {e}")
                        break
            
            async def receive_transcription():
                global transcription_text, detected_language
                while is_running:
                    try:
                        response = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                        if response.get("type") == "transcription.delta":
                            transcription_text += response.get("delta", "")
                            if "language" in response:
                                detected_language = response["language"]
                            update_callback(transcription_text, detected_language)
                        elif response.get("type") == "transcription.done":
                            transcription_text = response.get("text", transcription_text)
                            update_callback(transcription_text, detected_language)
                        elif response.get("type") == "error":
                            print(f"❌ Error: {response.get('error')}")
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if is_running:
                            print(f"Error recibiendo: {e}")
                        break
            await asyncio.gather(send_audio(), receive_transcription())
    except Exception as e:
        print(f"❌ Error de conexión WebSocket: {e}")


def run_websocket_loop(update_callback):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_handler(update_callback))


def process_audio(audio_data, transcription, translation, lang):
    global is_running, transcription_text, detected_language
    if audio_data is None:
        return transcription, translation, lang
    sample_rate, audio_array = audio_data
    if len(audio_array.shape) > 1:
        audio_array = audio_array.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        try:
            import soxr
            audio_array = soxr.resample(audio_array.astype(np.float32), sample_rate, SAMPLE_RATE)
        except ImportError:
            ratio = SAMPLE_RATE / sample_rate
            new_length = int(len(audio_array) * ratio)
            audio_array = np.interp(np.linspace(0, len(audio_array), new_length), np.arange(len(audio_array)), audio_array)
    if audio_array.dtype in [np.float32, np.float64]:
        audio_array = (audio_array * 32767).astype(np.int16)
    elif audio_array.dtype != np.int16:
        audio_array = audio_array.astype(np.int16)
    audio_bytes = audio_array.tobytes()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    if is_running:
        audio_queue.put(audio_base64)
    current_transcription = transcription_text
    current_lang = detected_language
    current_translation = ""
    if current_transcription.strip():
        current_translation = translate_text(current_transcription, current_lang, "en")
    return current_transcription, current_translation, f"Idioma: {current_lang}"


def start_recording():
    global is_running, ws_thread, transcription_text, detected_language
    if is_running:
        return "⏹️ Ya está grabando...", "", "", "Idioma: es"
    is_running = True
    transcription_text = ""
    detected_language = "es"
    def update_callback(text, lang):
        pass
    ws_thread = threading.Thread(target=run_websocket_loop, args=(update_callback,), daemon=True)
    ws_thread.start()
    return "🎤 Grabando... (habla ahora)", "", "", "Idioma: es"


def stop_recording():
    global is_running, transcription_text, detected_language
    is_running = False
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except:
            break
    final_transcription = transcription_text
    final_lang = detected_language
    final_translation = translate_text(final_transcription, final_lang, "en") if final_transcription else ""
    return f"✅ Detenido. Idioma detectado: {final_lang}", final_transcription, final_translation, f"Idioma: {final_lang}"


with gr.Blocks(title="🎤 Transcripción + Traducción en Tiempo Real") as demo:
    gr.Markdown("""
    # 🎤 Transcripción y Traducción en Tiempo Real
    
    **Modelos utilizados:**
    - **Transcripción**: Voxtral-Mini-4B-Realtime (Mistral AI) - 13 idiomas
    - **Traducción**: NLLB-200-distilled-600M (Meta) - 200+ idiomas
    
    ---
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            status = gr.Textbox(label="Estado", value="⏸️ Listo para grabar", interactive=False)
            with gr.Row():
                start_btn = gr.Button("🎤 Iniciar", variant="primary", size="lg")
                stop_btn = gr.Button("⏹️ Detener", variant="stop", size="lg")
            audio_input = gr.Audio(sources=["microphone"], streaming=True, label="Micrófono", type="numpy")
            lang_display = gr.Textbox(label="Idioma Detectado", value="Idioma: es", interactive=False)
    
    with gr.Row():
        with gr.Column():
            transcription_output = gr.Textbox(label="📝 Transcripción (idioma original)", placeholder="La transcripción aparecerá aquí...", lines=8, max_lines=15, interactive=False)
        with gr.Column():
            translation_output = gr.Textbox(label="🌍 Traducción (Inglés)", placeholder="La traducción aparecerá aquí...", lines=8, max_lines=15, interactive=False)
    
    start_btn.click(fn=start_recording, inputs=[], outputs=[status, transcription_output, translation_output, lang_display])
    stop_btn.click(fn=stop_recording, inputs=[], outputs=[status, transcription_output, translation_output, lang_display])
    audio_input.stream(fn=process_audio, inputs=[audio_input, transcription_output, translation_output, lang_display], outputs=[transcription_output, translation_output, lang_display])
    
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
    ╚══════════════════════════════════════════════════════════════╝
    """)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
