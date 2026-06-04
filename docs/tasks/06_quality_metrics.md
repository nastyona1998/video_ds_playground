# Задание 06: Метрики качества речи

## Цель

Научиться вычислять инструментальные метрики качества речи: **PESQ**, **STOI**, **SNR**, **акцент** — и интерпретировать их применительно к задачам транскрибации.

---

## Теоретическая справка

### PESQ (Perceptual Evaluation of Speech Quality)

Стандарт ITU-T P.862. Имитирует субъективную оценку качества речи слушателем. Диапазон: **-0.5 ... 4.5** (MOS-LQO).

- `< 1.0` — неприемлемое качество
- `2.0–3.0` — удовлетворительное
- `> 3.5` — хорошее

Требует **эталонный сигнал** (clean reference). Используется для оценки эффекта шумоподавления.

- [PESQ в torchaudio](https://pytorch.org/audio/stable/generated/torchaudio.functional.pesq.html)
- [ITU-T P.862](https://www.itu.int/rec/T-REC-P.862)

### STOI (Short-Time Objective Intelligibility)

Оценка разборчивости речи. Диапазон: **0 ... 1**.

- `< 0.5` — плохая разборчивость
- `> 0.8` — хорошая разборчивость

Коррелирует с процентом правильно распознанных слов.

- [STOI в torchaudio](https://pytorch.org/audio/stable/generated/torchaudio.functional.stoi.html)

### SNR (Signal-to-Noise Ratio)

Отношение сигнал/шум в децибелах. Чем выше, тем меньше шума.

```python
snr_db = 10 * log10(signal_power / noise_power)
```

- `> 20 dB` — чистая речь
- `10–20 dB` — умеренный шум
- `< 10 dB` — сильный шум

### Определение акцента

Модели классификации акцента (например, на базе wav2vec2) предсказывают родной язык говорящего. Полезно для выбора языковой модели транскрибации.

- [accent-classification на HuggingFace](https://huggingface.co/models?search=accent)

---

## Задание

### Шаг 1. Изучите `QualityMetricsStage`

Файл: `src/stages/quality_metrics.py`

### Шаг 2. Реализуйте функции в `src/utils/metrics.py`

```python
def compute_pesq(reference_path: str, degraded_path: str, 
                 sample_rate: int = 16000, mode: str = "wb") -> float:
    """
    Вычисляет PESQ.
    mode: "wb" (wideband, 16kHz) или "nb" (narrowband, 8kHz)
    """
    # TODO: используйте torchaudio.functional.pesq()
    pass

def compute_stoi(reference_path: str, degraded_path: str,
                 sample_rate: int = 16000, extended: bool = False) -> float:
    """Вычисляет STOI."""
    # TODO: используйте torchaudio.functional.stoi()
    pass

def compute_snr(audio_path: str) -> float:
    """Оценивает SNR без эталонного сигнала (blind SNR estimation)."""
    # TODO: используйте VAD-маску для разделения речи и шума
    pass
```

### Шаг 3. Запустите на нескольких аудио

Подготовьте:
- `clean.wav` — чистая речь (или запись в тихой комнате)
- `noisy.wav` — та же запись с добавленным шумом (или реальная шумная запись)

```bash
python run.py --config config/default_pipeline.yaml \
  --video data/raw_videos/sample.mp4 \
  --stages audio_extract,quality_metrics
```

### Шаг 4. Проанализируйте корреляцию с WER

После этапа транскрибации сравните: есть ли корреляция между PESQ/STOI и WER?

---

## Ожидаемый результат

```json
{
  "metrics": {
    "pesq": 2.87,
    "stoi": 0.823,
    "snr_db": 18.4,
    "accent_score": {"ru": 0.92, "en": 0.05, "other": 0.03},
    "duration_sec": 3612.5,
    "speech_ratio": 0.847
  }
}
```

---

## Дополнительные вызовы

- Сгенерируйте синтетически зашумлённые версии аудио с разным SNR (0, 5, 10, 20 дБ) и постройте графики PESQ/STOI vs SNR.
- Исследуйте, при каком значении PESQ WER резко возрастает.
- Реализуйте оценку **MOS** (Mean Opinion Score) через нейросетевую модель (например, DNSMOS).
