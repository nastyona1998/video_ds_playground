# Задание 08: Постобработка транскрибации и борьба с галлюцинациями

## Цель

Реализовать систему постобработки транскрипции: удаление галлюцинаций, повторений, служебных фраз (водяных знаков субтитров), нормализацию текста.

---

## Теоретическая справка

### Проблема галлюцинаций Whisper

Whisper — генеративная модель. На участках тишины или шума она **галлюцинирует** — генерирует несуществующий текст. Типичные паттерны:

```
"Автор и редактор субтитров Dimatorzok"
"Субтитры делал Ваня"
"Продолжение следует..."
"Спасибо за просмотр!"
"[музыка]"
"[аплодисменты]"
"..."
```

Методы обнаружения:
1. **Регулярные выражения** — список известных паттернов
2. **Анализ log-probability** — низкая уверенность Whisper = подозрительный сегмент
3. **VAD-верификация** — если в сегменте нет речи (по VAD), но есть текст → галлюцинация
4. **Повторения** — одна и та же фраза 3+ раза подряд

### Проблема повторений (looping)

Whisper иногда "зависает" и повторяет одну фразу много раз подряд:

```
"да да да да да да да да да да"
"конечно конечно конечно конечно"
```

### Нормализация текста

После транскрибации часто нужно:
- Расставить пунктуацию (если модель её не ставит)
- Нормализовать числа: "два тысячи двадцать четыре" → "2024"
- Убрать лишние пробелы, странные символы

- [clean-text](https://github.com/jfilter/clean-text)
- [spacy](https://spacy.io/) — для NLP постобработки
- [whisper_hallucination_cleaner](https://github.com/yoavram/whisper-hallucination-cleaner)

---

## Задание

### Шаг 1. Изучите `PostprocessStage`

Файл: `src/stages/postprocess.py`

### Шаг 2. Реализуйте `HallucinationDetector`

```python
# src/utils/text_cleaner.py

# Известные паттерны галлюцинаций
HALLUCINATION_PATTERNS = [
    r"[Аа]втор\s+(и\s+)?редактор\s+субтитров?\s+\w+",
    r"[Сс]убтитры\s+(делал|создал|от)\s+\w+",
    r"[Пп]родолжение\s+следует",
    r"[Сс]пасибо\s+за\s+просмотр",
    r"[Пп]одпишитесь\s+на\s+канал",
    r"\[музыка\]",
    r"\[аплодисменты\]",
    r"\[смех\]",
    r"\.{3,}",             # "........"
    r"(?i)subscribe",
    r"(?i)like\s+and\s+subscribe",
]

def detect_hallucinations(segments: list[dict]) -> list[dict]:
    """
    Проверяет каждый сегмент на галлюцинации.
    Возвращает список с флагом is_hallucination.
    """
    # TODO: применить HALLUCINATION_PATTERNS к каждому сегменту
    pass

def detect_repetitions(text: str, min_repeat: int = 3) -> list[str]:
    """
    Находит повторяющиеся фразы (>= min_repeat раз подряд).
    """
    # TODO: регулярное выражение или алгоритм скользящего окна
    pass
```

### Шаг 3. Реализуйте VAD-верификацию галлюцинаций

```python
def verify_with_vad(segments: list[dict], 
                    speech_intervals: list[dict],
                    min_overlap: float = 0.3) -> list[dict]:
    """
    Помечает сегмент как галлюцинацию, если он перекрывается
    со speech_intervals менее чем на min_overlap.
    """
    # TODO: проверить overlap каждого текстового сегмента
    # с интервалами речи от VAD
    pass
```

### Шаг 4. Запустите

```bash
python run.py --config config/default_pipeline.yaml \
  --video data/raw_videos/sample.mp4 \
  --stages fetch,audio_extract,vad,transcription,postprocess
```

---

## Ожидаемый результат

```json
{
  "clean_transcription": "Добрый день. Сегодня мы рассмотрим тему машинного обучения...",
  "removed_fragments": [
    {
      "text": "Автор субтитров Dimatorzok",
      "reason": "hallucination_pattern",
      "start": 3612.0,
      "end": 3614.5
    },
    {
      "text": "да да да да да да",
      "reason": "repetition",
      "start": 1205.3,
      "end": 1208.1
    }
  ],
  "stats": {
    "total_segments": 847,
    "removed_segments": 12,
    "removed_ratio": 0.014
  }
}
```

---

## Дополнительные вызовы

- Пополните `HALLUCINATION_PATTERNS` паттернами, найденными в реальных транскрипциях.
- Реализуйте фильтрацию по `avg_logprob` из Whisper-сегментов (порог: `< -1.0` — подозрительно).
- Добавьте расстановку знаков препинания с помощью [deepmultilingualpunctuation](https://github.com/oliverguhr/deepmultilingualpunctuation).
- Сравните WER транскрипции до и после постобработки.
