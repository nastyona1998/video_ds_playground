# Задание 07: Шумоподавление и его влияние на WER

## Цель

Применить несколько методов шумоподавления и **экспериментально оценить** их влияние на WER разных ASR-моделей. Ключевой вывод: сильные модели (Whisper large) устойчивы к шуму и могут ухудшиться от агрессивного шумоподавления; слабые модели выигрывают от предварительной очистки.

---

## Теоретическая справка

### Почему шумоподавление не всегда помогает ASR?

Whisper large-v3 обучен на зашумлённых данных — он **уже умеет** работать с шумом. Агрессивное шумоподавление:
- искажает спектральные характеристики речи
- создаёт "музыкальный шум" (musical noise artifacts)
- может удалять тихие согласные

→ В результате WER для Whisper large **увеличивается** после обработки.

Для слабых моделей (wav2vec2 без файнтюнинга на шумных данных):
- шум разрушает CTC-выравнивание
- шумоподавление **уменьшает** WER

### Методы шумоподавления

| Метод | Подход | Скорость | Качество |
|-------|--------|----------|----------|
| **noisereduce** | Спектральное вычитание | мгновенно | среднее |
| **SpeechBrain** | DNN (SEGAN, ConvTasNet) | медленно | высокое |
| **demucs** | U-Net / трансформер | медленно | очень высокое |
| **RNNoise** | Нейросеть (C, реалтайм) | мгновенно | хорошее |

- [noisereduce](https://github.com/timsainburg/noisereduce)
- [SpeechBrain](https://speechbrain.github.io/) — [pretrained models](https://huggingface.co/speechbrain)
- [demucs](https://github.com/facebookresearch/demucs)
- [RNNoise](https://jmvalin.ca/demo/rnnoise/)

### Спектральное вычитание (noisereduce)

```
S_clean(f) = S_noisy(f) - S_noise(f)
```

Оценка шума: первые 0.5 секунды записи (предполагается тишина).

---

## Задание

### Шаг 1. Реализуйте `NoiseReduceWrapper`

```python
# src/models/noisereduce_wrapper.py
import noisereduce as nr
import numpy as np
import soundfile as sf

@register_model("noisereduce")
class NoiseReduceWrapper(ModelWrapper):
    def load(self, **kwargs):
        pass  # нет весов для загрузки

    def predict(self, audio_path: str, prop_decrease: float = 1.0,
                stationary: bool = False, **kwargs) -> dict:
        """
        prop_decrease: 0.0 = без изменений, 1.0 = максимальное подавление
        stationary: True = стационарный шум, False = нестационарный
        """
        # TODO: загрузите аудио, примените nr.reduce_noise(), сохраните
        pass
```

### Шаг 2. Реализуйте `SpeechBrainDenoiser`

```python
# src/models/speechbrain_wrapper.py
from speechbrain.pretrained import SepformerSeparation

@register_model("speechbrain-sepformer")
class SpeechBrainDenoiser(ModelWrapper):
    def load(self, **kwargs):
        self.model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wham",
            savedir="data/.cache/speechbrain"
        )

    def predict(self, audio_path: str, **kwargs) -> dict:
        # TODO: инференс, сохранение очищенного аудио
        pass
```

### Шаг 3. Запустите сравнительный эксперимент

```bash
python src/experiments/denoise_impact.py \
  --denoisers noisereduce,speechbrain-sepformer,demucs \
  --transcribers whisper-small,whisper-large-v3,wav2vec2-xlsr-ru \
  --video data/raw_videos/noisy_sample.mp4 \
  --reference data/transcripts/sample_ground_truth.txt
```

---

## Ожидаемый результат

```
Эксперимент: влияние шумоподавления на WER
Аудио: noisy_sample.wav (SNR ≈ 8 dB)

┌─────────────────────────┬───────────────┬───────────────┬──────────────┐
│ Конфигурация            │ whisper-small │ whisper-large │ wav2vec2-ru  │
├─────────────────────────┼───────────────┼───────────────┼──────────────┤
│ Без обработки           │    0.312      │    0.089      │    0.487     │
│ + noisereduce (mild)    │    0.278      │    0.091      │    0.341     │
│ + noisereduce (max)     │    0.265      │    0.112      │    0.298     │  ← large ухудшился!
│ + speechbrain-sepformer │    0.241      │    0.094      │    0.267     │
│ + demucs                │    0.233      │    0.096      │    0.251     │
└─────────────────────────┴───────────────┴───────────────┴──────────────┘
```

---

## Дополнительные вызовы

- Послушайте аудио до и после шумоподавления — субъективно лучше или нет?
- Постройте график: WER vs уровень шума (SNR) для каждой модели.
- Попробуйте применить шумоподавление **только** к тихим участкам (по VAD-маске).
- Реализуйте адаптивный выбор: если SNR < 15 дБ — применять шумоподавление, иначе нет.
