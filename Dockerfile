FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Системные зависимости
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    ffmpeg \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Сначала устанавливаем тяжёлые зависимости (кешируются)
RUN pip install --upgrade pip
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

# spacy модель
RUN python -m spacy download ru_core_news_sm || true

COPY . .
RUN pip install -e .

# Проверяем что ffmpeg доступен
RUN ffmpeg -version | head -1

VOLUME ["/app/data"]

ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
