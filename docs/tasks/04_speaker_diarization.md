# Задание 04: Диаризация спикеров

## Цель

Определить, **кто** и **когда** говорит в аудиозаписи. Использовать таймкоды слов из транскрибации для разметки текста по спикерам.

---

## Теоретическая справка

**Диаризация** (Speaker Diarization) — задача разбиения аудио на сегменты, принадлежащие разным говорящим. Ответ на вопрос: *"Кто говорил когда?"*

### Методы

| Метод | Описание |
|-------|----------|
| **pyannote.audio** | SOTA нейросетевая диаризация (requires HF token) |
| **simple_diarizer** | Упрощённая диаризация на базе resemblyzer |
| **Объединение с ASR** | Наложение таймкодов слов из Whisper на сегменты диаризации |

- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [pyannote paper](https://arxiv.org/abs/2104.04045)
- [simple_diarizer](https://github.com/cvqluu/simple_diarizer)
- [WhisperX](https://github.com/m-bain/whisperX) — комбинирует Whisper + диаризацию

### Алгоритм объединения с таймкодами слов

```
Whisper segments:  [{"start": 0.5, "end": 1.2, "word": "Привет"}]
Diarization:       [{"start": 0.0, "end": 3.5, "speaker": "SPEAKER_00"}]

→ Для каждого слова найти сегмент диаризации с максимальным перекрытием
→ Результат: [{"word": "Привет", "speaker": "SPEAKER_00", "start": 0.5}]
```

---

## Задание

### Шаг 1. Изучите `DiarizationStage`

Файл: `src/stages/speaker_diarization.py`

### Шаг 2. Реализуйте `PyannoteWrapper`

```python
# src/models/pyannote_wrapper.py
from pyannote.audio import Pipeline as PyannotePipeline

@register_model("pyannote")
class PyannoteWrapper(ModelWrapper):
    def load(self, hf_token: str = None, **kwargs):
        self.pipeline = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )

    def predict(self, audio_path: str, num_speakers: int = None, **kwargs):
        result = self.pipeline(audio_path, num_speakers=num_speakers)
        segments = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return {"segments": segments}
```

### Шаг 3. Реализуйте функцию объединения

В `src/stages/speaker_diarization.py` реализуйте `_merge_with_transcription()`:
- Принимает `word_timestamps` из Whisper и `speaker_turns` из диаризации
- Возвращает список слов с полем `speaker`

### Шаг 4. Запустите

```bash
python run.py --config config/default_pipeline.yaml \
  --video data/raw_videos/sample.mp4 \
  --stages fetch,audio_extract,transcription,diarization
```

---

## Ожидаемый результат

```json
{
  "speaker_transcription": [
    {"speaker": "SPEAKER_00", "start": 0.5, "end": 12.3, "text": "Добрый день, сегодня мы рассмотрим..."},
    {"speaker": "SPEAKER_01", "start": 12.8, "end": 25.1, "text": "Спасибо за вопрос..."}
  ]
}
```

---

## Дополнительные вызовы

- Попробуйте `simple_diarizer` без HuggingFace токена. Сравните точность.
- Реализуйте подсчёт времени речи каждого спикера.
- Визуализируйте диаризацию в виде цветной временной шкалы (см. `src/utils/report_generator.py`).
