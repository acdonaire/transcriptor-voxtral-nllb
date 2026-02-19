import os, asyncio, base64, json, queue, threading, time
import gradio as gr
import numpy as np
import websockets
import httpx

VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
NLLB_HOST = os.environ.get("NLLB_HOST", "localhost")
NLLB_PORT = os.environ.get("NLLB_PORT", "8001")
SAMPLE_RATE = 16_000
MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"

# Buffer config
DEBOUNCE_SECONDS = 1.5
MIN_NEW_CHARS = 10

LANGUAGES = {
    "Inglés": "en",
    "Español": "es", 
    "Francés": "fr",
    "Alemán": "de",
    "Italiano": "it",
    "Portugués": "pt",
    "Holandés": "nl",
    "Ruso": "ru",
    "Chino": "zh",
    "Japonés": "ja",
    "Coreano": "ko",
    "Árabe": "ar",
}

audio_queue = queue.Queue()
transcription_text = ""
detected_language = "es"
is_running = False

# Buffer state
last_translation_time = 0
last_translated_text = ""
current_translation = ""

def translate_text(text, src="es", tgt="en"):
    if not text.strip() or src == tgt:
        return text if src == tgt else ""
    try:
        r = httpx.post(f"http://{NLLB_HOST}:{NLLB_PORT}/translate", json={"text": text, "source_lang": src, "target_lang": tgt}, timeout=30)
        return r.json().get("translation", "") if r.status_code == 200 else ""
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def should_translate(current_text):
    """Decide si debemos traducir basado en debounce y cambio mínimo."""
    global last_translation_time, last_translated_text
    
    now = time.time()
    time_since_last = now - last_translation_time
    new_chars = len(current_text) - len(last_translated_text)
    
    # Traducir si: pasó suficiente tiempo Y hay suficiente texto nuevo
    if time_since_last >= DEBOUNCE_SECONDS and new_chars >= MIN_NEW_CHARS:
        return True
    return False

def update_translation(text, src, tgt):
    """Actualiza la traducción si cumple condiciones del buffer."""
    global last_translation_time, last_translated_text, current_translation
    
    if should_translate(text):
        current_translation = translate_text(text, src, tgt)
        last_translation_time = time.time()
        last_translated_text = text
    
    return current_translation

async def ws_handler():
    global transcription_text, is_running, detected_language
    ws_url = f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime"
    print(f"Conectando a {ws_url}...")
    try:
        async with websockets.connect(ws_url) as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            print("Sesión WebSocket iniciada")
            
            async def send():
                while is_running:
                    try:
                        chunk = await asyncio.get_event_loop().run_in_executor(None, lambda: audio_queue.get(timeout=0.1))
                        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
                    except queue.Empty:
                        continue
                    except:
                        break
            
            async def recv():
                global transcription_text, detected_language
                async for message in ws:
                    if not is_running:
                        break
                    data = json.loads(message)
                    msg_type = data.get("type")
                    if msg_type == "transcription.delta":
                        transcription_text += data.get("delta", "")
                        detected_language = data.get("language", detected_language)
                    elif msg_type == "transcription.done":
                        transcription_text = data.get("text", transcription_text)
            
            await asyncio.gather(send(), recv())
    except Exception as e:
        print(f"WS error: {e}")

def run_ws():
    asyncio.new_event_loop().run_until_complete(ws_handler())

def process_audio(audio, trans, trad, lang, target_lang):
    global is_running, transcription_text, detected_language
    if audio is None or not is_running:
        return trans, trad, lang
    
    sr, arr = audio
    
    if len(arr.shape) > 1:
        arr = arr.mean(axis=1)
    
    if arr.dtype == np.int16:
        audio_float = arr.astype(np.float32) / 32767.0
    else:
        audio_float = arr.astype(np.float32)
    
    if sr != SAMPLE_RATE:
        num_samples = int(len(audio_float) * SAMPLE_RATE / sr)
        audio_float = np.interp(
            np.linspace(0, len(audio_float) - 1, num_samples),
            np.arange(len(audio_float)),
            audio_float
        )
    
    pcm16 = (audio_float * 32767).astype(np.int16)
    audio_queue.put(base64.b64encode(pcm16.tobytes()).decode())
    
    t, l = transcription_text, detected_language
    tgt = LANGUAGES.get(target_lang, "en")
    
    # Usar buffer para traducción
    translation = update_translation(t, l, tgt) if t.strip() else ""
    
    return t, translation, f"Idioma: {l}"

def start():
    global is_running, transcription_text, detected_language
    global last_translation_time, last_translated_text, current_translation
    
    # Limpiar todo al iniciar
    is_running = False  # Parar cualquier grabación anterior
    time.sleep(0.1)
    
    # Limpiar cola de audio
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break
    
    # Reset estado
    transcription_text = ""
    detected_language = "es"
    last_translation_time = 0
    last_translated_text = ""
    current_translation = ""
    
    # Iniciar nueva grabación
    is_running = True
    threading.Thread(target=run_ws, daemon=True).start()
    
    return "🎤 Grabando...", "", "", "Idioma: es"

def stop(target_lang):
    global is_running, transcription_text, detected_language, current_translation
    is_running = False
    
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break
    
    t, l = transcription_text, detected_language
    tgt = LANGUAGES.get(target_lang, "en")
    
    # Traducción final completa
    final_translation = translate_text(t, l, tgt) if t else ""
    current_translation = final_translation
    
    return f"✅ Detenido ({l})", t, final_translation, f"Idioma: {l}"

with gr.Blocks(title="Transcripción + Traducción") as demo:
    gr.Markdown("# 🎤 Transcripción y Traducción en Tiempo Real\n**Voxtral + NLLB**\n---")
    with gr.Row():
        with gr.Column():
            status = gr.Textbox(label="Estado", value="⏸️ Listo", interactive=False)
            target_lang = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="Inglés",
                label="🌍 Traducir a"
            )
            with gr.Row():
                start_btn = gr.Button("🎤 Iniciar", variant="primary")
                stop_btn = gr.Button("⏹️ Detener", variant="stop")
            audio_input = gr.Audio(sources=["microphone"], streaming=True, label="Micrófono", type="numpy")
            lang_display = gr.Textbox(label="Idioma detectado", value="Idioma: es", interactive=False)
    with gr.Row():
        trans_out = gr.Textbox(label="📝 Transcripción", lines=8, interactive=False)
        trad_out = gr.Textbox(label="🌍 Traducción", lines=8, interactive=False)
    
    start_btn.click(start, [], [status, trans_out, trad_out, lang_display])
    stop_btn.click(stop, [target_lang], [status, trans_out, trad_out, lang_display])
    audio_input.stream(process_audio, [audio_input, trans_out, trad_out, lang_display, target_lang], [trans_out, trad_out, lang_display])

if __name__ == "__main__":
    print(f"VLLM: ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime")
    print(f"NLLB: http://{NLLB_HOST}:{NLLB_PORT}/translate")
    print(f"Buffer: {DEBOUNCE_SECONDS}s debounce, {MIN_NEW_CHARS} chars min")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
