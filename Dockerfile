FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HF_HOME=/home/site/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/site/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/site/.cache/sentence_transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

RUN mkdir -p /app/uploads

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "--timeout", "300", "app:app"]
