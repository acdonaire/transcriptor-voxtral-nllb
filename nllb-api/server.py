from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import uvicorn

app = FastAPI()
LANG_TO_NLLB = {"es": "spa_Latn", "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "ru": "rus_Cyrl", "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "ar": "arb_Arab", "hi": "hin_Deva"}
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)
print(f"NLLB cargado en {device}")

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "es"
    target_lang: str = "en"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/translate")
def translate(req: TranslationRequest):
    if not req.text.strip() or req.source_lang == req.target_lang:
        return {"translation": req.text if req.source_lang == req.target_lang else "", "source_lang": req.source_lang, "target_lang": req.target_lang}
    tokenizer.src_lang = LANG_TO_NLLB.get(req.source_lang, "spa_Latn")
    inputs = tokenizer(req.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_TO_NLLB.get(req.target_lang, "eng_Latn")), max_new_tokens=256, num_beams=4)
    return {"translation": tokenizer.decode(outputs[0], skip_special_tokens=True), "source_lang": req.source_lang, "target_lang": req.target_lang}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
