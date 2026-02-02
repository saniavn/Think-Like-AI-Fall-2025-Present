FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip first and then install requirements with PyTorch CPU wheels index correctly formatted (no Markdown)
RUN pip install --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/cpu

# Preload transformers models at build time to speed up container startup
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
print('Downloading SmolLM-360M...'); \
AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-360M-Instruct'); \
AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-360M-Instruct'); \
print('Model download complete.')"

COPY . .

EXPOSE 8080

CMD exec gunicorn -b :${PORT:-8080} --workers 1 --timeout 600 app:app
