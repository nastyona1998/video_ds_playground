# Задание 03: Транскрибация и сравнение WER

## Цель

Реализовать транскрибацию аудио с помощью нескольких моделей (Whisper, faster-whisper, wav2vec2) и сравнить их по метрике **WER (Word Error Rate)**.

---

## Теоретическая справка

### Word Error Rate (WER)

WER — основная метрика качества ASR-систем:

```
WER = (S + D + I) / N
```

где:
- `S` — замены (substitutions)
- `D` — удаления (deletions)
- `I` — вставки (insertions)
- `N` — количество слов в эталоне

WER = 0.0 — идеальное совпадение. WER > 1.0 — модель хуже случайного угадывания.

- [jiwer — библиотека для WER](https://github.com/jitsi/jiwer)
- [OpenAI Whisper paper](https://arxiv.org/abs/2212.04356)
- [Wav2Vec2 paper](https://arxiv.org/abs/2006.11477)

### Whisper

OpenAI Whisper — энкодер-декодер на трансформерах, обученный на 680 000 часов мультиязычных данных. Устойчив к шуму, поддерживает 99 языков.

Размеры моделей:

| Модель        | Параметры | VRAM  | Скорость |
|---------------|-----------|-------|----------|
| whisper-tiny  | 39M       | ~1 GB | быстро   |
| whisper-base  | 74M       | ~1 GB | быстро   |
| whisper-small | 244M      | ~2 GB | средне   |
| whisper-medium| 769M      | ~5 GB | медленно |
| whisper-large-v3 | 1550M  | ~10 GB| медленно |

- [openai-whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2, в 2–4× быстрее

### Wav2Vec2

Meta Wav2Vec2 — CTC-модель, обученная на самонадзоре. Хорошо работает для конкретных языков при файнтюнинге.

- [HuggingFace Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53)

---

## Структура кода

- `src/stages/transcription.py` — этап транскрибации
- `src/models/whisper_wrapper.py` — обёртка для Whisper
- `src/models/wav2vec2_wrapper.py` — **ваша задача**
- `src/utils/metrics.py` — функция `compute_wer()`
- `src/experiments/compare_transcribers.py` — скрипт сравнения

---

## Задание

### Шаг 1. Изучите `WhisperWrapper`

```python
# src/models/whisper_wrapper.py
@register_model("whisper-small")
@register_model("whisper-large-v3")
class WhisperWrapper(ModelWrapper):
    def load(self, model_size: str = "small", device: str = "cpu"):
        self.model = whisper.load_model(model_size, device=device)

    def predict(self, audio_path: str, language: str = None, **kwargs) -> dict:
        result = self.model.transcribe(audio_path, language=language)
        return {
            "text": result["text"],
            "segments": result["segments"],   # [{start, end, text, words}]
            "language": result["language"]
        }
```

### Шаг 2. Реализуйте `Wav2Vec2Wrapper`

Создайте файл `src/models/wav2vec2_wrapper.py`:

```python
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch, torchaudio
from src.core.base import ModelWrapper
from src.core.registry import register_model

@register_model("wav2vec2-xlsr-ru")
class Wav2Vec2Wrapper(ModelWrapper):
    def load(self, model_name: str = "bond005/wav2vec2-large-ru-golos", device: str = "cpu"):
        # TODO: загрузите processor и model
        pass

    def predict(self, audio_path: str, **kwargs) -> dict:
        # TODO: предобработка аудио, инференс, декодирование
        # Возвращайте {"text": str, "segments": [], "language": "ru"}
        pass
```

**Подсказка**: Wav2Vec2 принимает тензор формы `[1, T]` с нормализованными значениями `float32`. Используйте `torchaudio.load()`.

### Шаг 3. Запустите сравнение

```bash
python src/experiments/compare_transcribers.py \
  --models whisper-small,whisper-large-v3,wav2vec2-xlsr-ru \
  --video data/raw_videos/sample.mp4 \
  --reference data/transcripts/sample_ground_truth.txt
```

### Шаг 4. Проанализируйте результаты

Изучите таблицу WER в отчёте. Ответьте на вопросы:
- Какая модель показала лучший WER?
- Как соотносятся WER и время инференса?
- При каком размере Whisper достигается хороший баланс?

---

## Ожидаемый результат

```
┌──────────────────────┬────────┬───────────┬────────────┐
│ Модель               │  WER   │ Время (с) │ Устройство │
├──────────────────────┼────────┼───────────┼────────────┤
│ whisper-tiny         │ 0.312  │    45     │   CPU      │
│ whisper-small        │ 0.187  │   123     │   CPU      │
│ whisper-large-v3     │ 0.098  │   580     │   GPU      │
│ wav2vec2-xlsr-ru     │ 0.215  │    90     │   CPU      │
└──────────────────────┴────────┴───────────┴────────────┘
```

---

## Дополнительные вызовы

- Попробуйте `faster-whisper` — насколько быстрее при том же WER?
- Реализуйте `FasterWhisperWrapper` в `src/models/faster_whisper_wrapper.py`.
- Сравните WER для русского и английского языков на одинаковых аудио.
- Добавьте метрику **CER (Character Error Rate)** в `src/utils/metrics.py`.
