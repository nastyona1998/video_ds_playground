# Задание 05: Детекция голосовой активности (VAD)

## Цель

Определить интервалы речи и тишины в аудио. Использовать VAD для предфильтрации перед транскрибацией и оценить его влияние на скорость работы и WER.

---

## Теоретическая справка

**VAD (Voice Activity Detection)** — задача бинарной классификации каждого аудио-фрейма: *"есть речь / нет речи"*.

### Применения VAD в пайплайне

1. **Ускорение транскрибации** — передавать ASR только участки с речью.
2. **Улучшение диаризации** — убирать тишину между репликами.
3. **Оценка длительности речи** — метрика "% речи в записи".
4. **Шумоподавление** — применять только к участкам с речью.

### Модели

| Модель | Описание | Точность | Скорость |
|--------|----------|----------|----------|
| **silero-vad** | ONNX, очень быстрый, хорошая точность | высокая | очень быстро |
| **pyannote VAD** | Нейросеть, SOTA | очень высокая | медленно |
| **webrtcvad** | Алгоритмический (Google), без GPU | средняя | мгновенно |

- [Silero VAD](https://github.com/snakers4/silero-vad)
- [pyannote VAD](https://huggingface.co/pyannote/voice-activity-detection)
- [webrtcvad](https://github.com/wiseman/py-webrtcvad)

### Параметры Silero VAD

```python
# threshold: 0.5 — порог уверенности модели
# min_silence_duration_ms: 100 — минимальная тишина (мс) для разделения
# min_speech_duration_ms: 250 — минимальная длина речевого сегмента
```

---

## Задание

### Шаг 1. Изучите `VADStage`

Файл: `src/stages/vad.py`

### Шаг 2. Реализуйте `SileroVADWrapper`

```python
# src/models/silero_vad.py
import torch

@register_model("silero")
class SileroVADWrapper(ModelWrapper):
    def load(self, **kwargs):
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad'
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils

    def predict(self, audio_path: str, threshold: float = 0.5,
                min_silence_ms: int = 100, min_speech_ms: int = 250, **kwargs):
        wav = self.read_audio(audio_path, sampling_rate=16000)
        timestamps = self.get_speech_timestamps(
            wav, self.model,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            min_speech_duration_ms=min_speech_ms,
            return_seconds=True
        )
        return {"speech_intervals": timestamps}
```

### Шаг 3. Сравните три VAD модели

Запустите скрипт:

```bash
python src/experiments/compare_vad.py \
  --models silero,webrtcvad,pyannote \
  --video data/raw_videos/sample.mp4
```

### Шаг 4. Измерьте ускорение транскрибации

Сравните Whisper:
1. Без VAD — на полном аудио
2. С VAD — только на речевых сегментах

---

## Ожидаемый результат

```json
{
  "speech_intervals": [
    {"start": 0.512, "end": 3.104},
    {"start": 4.832, "end": 12.416}
  ],
  "silence_intervals": [
    {"start": 3.104, "end": 4.832}
  ],
  "speech_ratio": 0.847,
  "total_speech_duration_sec": 3456.2
}
```

---

## Дополнительные вызовы

- Визуализируйте VAD-маску на аудиограмме (waveform + цветная маска).
- Сравните скорость транскрибации Whisper с VAD и без (на 60-минутном видео).
- Реализуйте адаптивный порог: если речи < 10% — предупреждать пользователя.
