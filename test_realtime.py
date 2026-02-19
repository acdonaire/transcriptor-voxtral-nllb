#!/usr/bin/env python3
"""Test cliente Voxtral Realtime - Basado en ejemplo oficial de Mistral"""

import asyncio
import base64
import json
import queue
import threading

import gradio as gr
import numpy as np
import websockets

SAMPLE_RATE = 16_000
WS_URL = "ws://localhost:8000/v1/realtime"
MODEL = "mistralai/Voxtral-Mini-4B-Realtime-2602"

audio_queue: queue.Queue = queue.Queue()
transcription_text = ""
is_running = False


async def websocket_handler():
    global transcription_text, is_running

    print(f"Conectando a {WS_URL}...")
    async with websockets.connect(WS_URL) as ws:
        response = await ws.recv()
        print(f"Recibido: {response[:100]}...")

        await ws.send(json.dumps({"type": "session.update", "model": MODEL}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("Sesión iniciada")

        async def send_audio():
            while is_running:
                try:
                    chunk = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: audio_queue.get(timeout=0.1)
                    )
                    await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
                except queue.Empty:
                    continue

        async def receive_transcription():
            global transcription_text
            async for message in ws:
                data = json.loads(message)
                print(f"Tipo: {data.get('type')}")
                if data.get("type") == "transcription.delta":
                    transcription_text += data["delta"]
                    print(f"Delta: {data['delta']}")

        await asyncio.gather(send_audio(), receive_transcription())


def start_websocket():
    global is_running
    is_running = True
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(websocket_handler())
    except Exception as e:
        print(f"WebSocket error: {e}")


def start_recording():
    global transcription_text
    transcription_text = ""
    thread = threading.Thread(target=start_websocket, daemon=True)
    thread.start()
    return gr.update(interactive=False), gr.update(interactive=True), ""


def stop_recording():
    global is_running
    is_running = False
    return gr.update(interactive=True), gr.update(interactive=False), transcription_text


def process_audio(audio):
    global transcription_text

    if audio is None or not is_running:
        return transcription_text

    sample_rate, audio_data = audio

    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.astype(np.float32)

    if sample_rate != SAMPLE_RATE:
        num_samples = int(len(audio_float) * SAMPLE_RATE / sample_rate)
        audio_float = np.interp(
            np.linspace(0, len(audio_float) - 1, num_samples),
            np.arange(len(audio_float)),
            audio_float,
        )

    pcm16 = (audio_float * 32767).astype(np.int16)
    b64_chunk = base64.b64encode(pcm16.tobytes()).decode("utf-8")
    audio_queue.put(b64_chunk)

    return transcription_text


with gr.Blocks(title="Test Voxtral Realtime") as demo:
    gr.Markdown("# 🎤 Test Voxtral Realtime\nCliente oficial de Mistral")

    with gr.Row():
        start_btn = gr.Button("Start", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop", interactive=False)

    audio_input = gr.Audio(sources=["microphone"], streaming=True, type="numpy")
    transcription_output = gr.Textbox(label="Transcription", lines=5)

    start_btn.click(start_recording, outputs=[start_btn, stop_btn, transcription_output])
    stop_btn.click(stop_recording, outputs=[start_btn, stop_btn, transcription_output])
    audio_input.stream(process_audio, inputs=[audio_input], outputs=[transcription_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
