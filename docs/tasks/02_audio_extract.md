# Задание 02: Извлечение аудио из видео (ffmpeg)

## Цель

Освоить работу с `ffmpeg` через Python для извлечения аудиодорожки из видеофайла, нормализации параметров (sample rate, каналы, формат) и получения метаданных медиафайла.

---

## Теоретическая справка

### ffmpeg

**ffmpeg** — мощный мультимедийный конвертер. Для задач ASR/транскрибации обычно нужно:
- Моно аудио (1 канал)
- Частота дискретизации 16 000 Hz (требование Whisper, wav2vec2)
- Формат WAV (PCM 16-bit)

- [ffmpeg официальная документация](https://ffmpeg.org/documentation.html)
- [ffmpeg-python на PyPI](https://pypi.org/project/ffmpeg-python/)

### Ключевые параметры ffmpeg для ASR

```bash
ffmpeg -i input.mp4 \
  -vn \           # без видео
  -acodec pcm_s16le \   # PCM 16-bit little-endian
  -ar 16000 \     # sample rate 16kHz
  -ac 1 \         # моно
  output.wav
```

### Метаданные

`ffprobe` — утилита из пакета ffmpeg для чтения метаданных:

```bash
ffprobe -v quiet -print_format json -show_streams input.mp4
```

- [ffprobe docs](https://ffmpeg.org/ffprobe.html)

---

## Структура кода

- `src/stages/audio_extractor.py` — этап пайплайна
- `src/utils/ffmpeg_helper.py` — вспомогательные функции

---

## Задание

### Шаг 1. Изучите `AudioExtractStage`

Откройте `src/stages/audio_extractor.py`. Метод `run()` принимает `data["video_path"]` и должен вернуть `data["audio_path"]` и `data["audio_meta"]`.

### Шаг 2. Реализуйте `extract_audio()` в `ffmpeg_helper.py`

```python
def extract_audio(
    video_path: str,
    output_path: str,
    sample_rate: int = 16000,
    channels: int = 1,
    audio_format: str = "wav"
) -> str:
    """
    Извлекает аудио из видеофайла.
    Возвращает путь к выходному файлу.
    """
    # TODO: реализуйте с помощью ffmpeg-python или subprocess
    pass
```

### Шаг 3. Реализуйте `get_media_info()` в `ffmpeg_helper.py`

Используйте `ffprobe` для получения длительности, кодека, sample rate оригинального файла.

### Шаг 4. Запустите этап

```bash
python run.py --config config/default_pipeline.yaml \
  --video data/raw_videos/sample.mp4 \
  --stages fetch,audio_extract
```

---

## Ожидаемый результат

```json
{
  "audio_path": "data/audio/sample_16k_mono.wav",
  "audio_meta": {
    "duration_sec": 3612.5,
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav",
    "original_codec": "aac",
    "original_sample_rate": 44100
  }
}
```

---

## Дополнительные вызовы

- Добавьте поддержку многоканального видео (5.1 → моно через downmix).
- Реализуйте нарезку аудио по временным интервалам (пригодится для обработки длинных видео по частям).
- Сравните качество WAV vs FLAC по размеру файла и проверьте, влияет ли формат на WER Whisper.
