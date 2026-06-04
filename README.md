# 🎓 Video Data Science Playground

Учебный полигон для освоения обработки видео и аудио в задачах Data Science.

## Что это такое?

**Video DS Playground** — это модульный Python-фреймворк и набор заданий, которые позволяют последовательно освоить полный пайплайн обработки видео:

```
Облако → Видео → Аудио → Транскрибация → Диаризация → VAD → Метрики → Шумоподавление → Постобработка
```

Вы читаете задание, изучаете теорию, реализуете класс, запускаете пайплайн и получаете отчёт с метриками.

---

## 🚀 Быстрый старт

### 1. Установка

```bash
git clone https://github.com/your-org/video_ds_playground.git
cd video_ds_playground

# Рекомендуется Python 3.10+
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
cp .env.example .env
# Откройте .env и заполните переменные (токен Nextcloud, HuggingFace и т.д.)
```

### 3. Запуск дефолтного пайплайна

```bash
python run.py --config config/default_pipeline.yaml --video /path/to/video.mp4
```

### 4. Сравнение моделей транскрибации

```bash
python src/experiments/compare_transcribers.py \
  --models whisper-small,whisper-large-v3,wav2vec2 \
  --video data/raw_videos/sample.mp4
```

### 5. Эксперимент: влияние шумоподавления

```bash
python src/experiments/denoise_impact.py \
  --denoisers noisereduce,speechbrain \
  --transcriber whisper-small \
  --video data/raw_videos/noisy_sample.mp4
```

---

## 📚 Задания (порядок прохождения)

| №  | Файл                              | Тема                                    |
|----|-----------------------------------|-----------------------------------------|
| 01 | [docs/tasks/01_cloud_fetch.md](docs/tasks/01_cloud_fetch.md)       | Загрузка видео из облака (Nextcloud/S3) |
| 02 | [docs/tasks/02_audio_extract.md](docs/tasks/02_audio_extract.md)   | Извлечение аудио через ffmpeg           |
| 03 | [docs/tasks/03_transcription.md](docs/tasks/03_transcription.md)   | Транскрибация и сравнение WER           |
| 04 | [docs/tasks/04_speaker_diarization.md](docs/tasks/04_speaker_diarization.md) | Диаризация спикеров             |
| 05 | [docs/tasks/05_vad.md](docs/tasks/05_vad.md)                       | Детекция голосовой активности (VAD)     |
| 06 | [docs/tasks/06_quality_metrics.md](docs/tasks/06_quality_metrics.md) | Метрики качества речи                |
| 07 | [docs/tasks/07_denoising.md](docs/tasks/07_denoising.md)           | Шумоподавление и его влияние на WER     |
| 08 | [docs/tasks/08_postprocessing.md](docs/tasks/08_postprocessing.md) | Постобработка и удаление галлюцинаций  |

---

## 🏗 Архитектура

```
src/
├── core/
│   ├── base.py           # Абстрактные классы Stage, ModelWrapper
│   ├── pipeline.py       # Класс Pipeline — выполняет этапы
│   ├── registry.py       # Реестр моделей и этапов
│   └── cloud_client.py   # Адаптеры облачных хранилищ
├── stages/               # Каждый этап — отдельный класс
├── models/               # Обёртки конкретных моделей
├── utils/                # Метрики, ffmpeg, постобработка, отчёты
└── experiments/          # Скрипты для сравнения методов
```

Подробнее: [docs/architecture.md](docs/architecture.md)

Как добавить свой этап: [docs/how_to_add_module.md](docs/how_to_add_module.md)

---

## 🐳 Docker (рекомендуется)

```bash
docker build -t video-ds-playground .
docker run --gpus all -v $(pwd)/data:/app/data video-ds-playground \
  python run.py --config config/default_pipeline.yaml --video data/raw_videos/sample.mp4
```

---

## 📊 Пример отчёта

После запуска эксперимента в папке `data/reports/` появится `report.html`:

- Сводная таблица WER / время инференса / использование шумоподавления
- Графики PESQ и STOI до/после фильтрации
- Фрагменты транскрипции с подсвеченными галлюцинациями
- Диаграмма диаризации спикеров по таймкодам

---

## 🛠 Технологический стек

| Задача                | Библиотеки |
|-----------------------|-----------|
| Облако                | `webdavclient3`, `boto3` |
| Аудио из видео        | `ffmpeg-python` |
| Транскрибация         | `openai-whisper`, `faster-whisper`, `transformers` |
| Диаризация            | `pyannote.audio`, `simple_diarizer` |
| VAD                   | `silero-vad`, `webrtcvad` |
| Метрики качества      | `torchaudio`, `librosa`, `torchmetrics` |
| Шумоподавление        | `noisereduce`, `speechbrain`, `demucs` |
| Постобработка         | `spacy`, `clean-text`, `regex` |
| Отчёты                | `pandas`, `matplotlib`, `jinja2`, `weasyprint` |

---

## 📜 Лицензия

MIT — используйте свободно в учебных и коммерческих проектах.
