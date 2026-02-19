import gradio as gr
import httpx

def translate(text, src, tgt):
    if not text.strip():
        return ""
    print(f"Traduciendo: '{text}' de '{src}' a '{tgt}'")
    try:
        r = httpx.post("http://localhost:8001/translate", 
                       json={"text": text, "source_lang": src, "target_lang": tgt}, 
                       timeout=30)
        result = r.json()
        print(f"Resultado: {result}")
        return result.get("translation", "Error") if r.status_code == 200 else f"Error: {r.status_code}"
    except Exception as e:
        return f"Error: {e}"

with gr.Blocks(title="Test NLLB") as demo:
    gr.Markdown("# Test NLLB-200 Translation")
    with gr.Row():
        src_lang = gr.Textbox(value="es", label="Origen (es, en, fr, de...)")
        tgt_lang = gr.Textbox(value="en", label="Destino (es, en, fr, de...)")
    input_text = gr.Textbox(label="Texto", lines=3, placeholder="Escribe aquí...")
    output_text = gr.Textbox(label="Traducción", lines=3)
    btn = gr.Button("Traducir", variant="primary")
    btn.click(translate, [input_text, src_lang, tgt_lang], output_text)

demo.launch(server_name="0.0.0.0", server_port=7862, share=True)
