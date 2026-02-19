FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')"
COPY server.py .
EXPOSE 8001
CMD ["python", "server.py"]
