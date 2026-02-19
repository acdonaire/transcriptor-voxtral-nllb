import os, asyncio, base64, json, queue, threading
import gradio as gr
import numpy as np
import websockets
import httpx

VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
NLLB_HOST = os.environ.get("NLLB_HOST", "localhost")
NLLB_PORT = os.environ.get("NLLB_PORT", "8001")
SAMPLE_RATE = 16_000

audio_queue = queue.Queue()
transcription_text = ""
detected_language = "es"
is_running = False

def translate_text(text, src="es", tgt="en"):
    if not text.strip() or src == tgt:
        return text if src == tgt else ""
    try:
        r = httpx.post(f"http://{NLLB_HOST}:{NLLB_PORT}/translate", json={"text": text, "source_lang": src, "target_lang": tgt}, timeout=30)
        return r.json().get("translation", "") if r.status_code == 200 else ""
    except:
        return ""

async def ws_handler():
    global transcription_text, is_running, detected_language
    try:
        async with websockets.connect(f"ws://{VLLM_HOST}:{VLLM_PORT}/v1/realtime") as ws:
            await ws.recv()
            await ws.send(json.dumps({"type": "session.update", "model": "mistralai/Voxtral-Mini-4B-Realtime-2602"}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
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
                while is_running:
                    try:
                        r = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.5))
                        if r.get("type") == "transcription.delta":
                            transcription_text += r.get("delta", "")
                            detected_language = r.get("language", detected_language)
                        elif r.get("type") == "transcription.done":
                            transcription_text = r.get("text", transcription_text)
                    except asyncio.TimeoutError:
                        continue
                    except:
                        break
            await asyncio.gather(send(), recv())
    except Exception as e:
        print(f"WS error: {e}")

def run_ws():
    asyncio.new_event_loop().run_until_complete(ws_handler())

def process_audio(audio, trans, trad, lang):
    global is_running, transcription_text, detected_language
    if audio is None:
        return trans, trad, lang
    sr, arr = audio
    if len(arr.shape) > 1:
        arr = arr.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import soxr
            arr = soxr.resample(arr.astype(np.float32), sr, SAMPLE_RATE)
        except:
            arr = np.interp(np.linspace(0, len(arr), int(len(arr)*SAMPLE_RATE/sr)), np.arange(len(arr)), arr)
    if arr.dtype in [np.float32, np.float64]:
        arr = (arr * 32767).astype(np.int16)
    if is_running:
        audio_queue.put(base64.b64encode(arr.tobytes()).decode())
    t, l = transcription_text, detected_language
    return t, translate_text(t, l, "en") if t.strip() else "", f"Idioma: {l}"

def start():
    global is_running, transcription_text, detected_language
    if is_running:
        return "Ya grabando...", "", "", "Idioma: es"
    is_running, transcription_text, detected_language = True, "", "es"
    threading.Thread(target=run_ws, daemon=True).start()
    return "🎤 Grabando...", "", "", "Idioma: es"

def stop():
    global is_running, transcription_text, detected_language
    is_running = False
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break
    t, l = transcription_text, detected_language
    return f"✅ Detenido ({l})", t, translate_text(t, l, "en") if t else "", f"Idioma: {l}"

with gr.Blocks(title="Transcripción + Traducción") as demo:
    gr.Markdown("# 🎤 Transcripción y Traducción\n**Voxtral + NLLB**\n---")
    with gr.Row():
        with gr.Column():
            status = gr.Textbox(label="Estado", value="⏸️ Listo", interactive=False)
            with gr.Row():
                start_btn = gr.Button("🎤 Iniciar", variant="primary")
                stop_btn = gr.Button("⏹️ Detener", variant="stop")
            audio_input = gr.Audio(sources=["microphone"], streaming=True, label="Micrófono", type="numpy")
            lang_display = gr.Textbox(label="Idioma", value="Idioma: es", interactive=False)
    with gr.Row():
        trans_out = gr.Textbox(label="📝 Transcripción", lines=8, interactive=False)
        trad_out = gr.Textbox(label="🌍 Traducción", lines=8, interactive=False)
    start_btn.click(start, [], [status, trans_out, trad_out, lang_display])
    stop_btn.click(stop, [], [status, trans_out, trad_out, lang_display])
    audio_input.stream(process_audio, [audio_input, trans_out, trad_out, lang_display], [trans_out, trad_out, lang_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
