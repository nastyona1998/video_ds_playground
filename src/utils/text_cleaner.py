"""
Постобработка транскрипции: удаление галлюцинаций, повторений, нормализация.

Задание (docs/tasks/08_postprocessing.md):
    Дополните HALLUCINATION_PATTERNS своими паттернами.
    Реализуйте функции detect_hallucinations() и detect_repetitions().
"""
from __future__ import annotations

import re
import unicodedata

# ------------------------------------------------------------------ #
#  Известные паттерны галлюцинаций Whisper                           #
# ------------------------------------------------------------------ #

HALLUCINATION_PATTERNS: list[str] = [
    # Водяные знаки субтитров
    r"[Аа]втор\s+(и\s+)?редактор\s+субтитров?\s+\w+",
    r"[Сс]убтитры\s+(делал|создал|написал|от|by)\s+\w+",
    r"[Пп]еревод\s+(субтитров?\s+)?\w+",
    r"[Тт]итры\s+(создал|от)\s+\w+",
    # Призывы к действию
    r"[Пп]одпишитесь\s+на\s+канал",
    r"[Нн]е\s+забудьте\s+подписаться",
    r"[Сс]тавьте\s+лайк",
    r"[Нн]ажмите\s+на\s+колокольчик",
    r"(?i)like\s+and\s+subscribe",
    r"(?i)don'?t\s+forget\s+to\s+subscribe",
    # Технические маркеры
    r"\[музыка\]",
    r"\[аплодисменты\]",
    r"\[смех\]",
    r"\[тишина\]",
    r"\[music\]",
    r"\[applause\]",
    r"\[laughter\]",
    r"(?i)\[inaudible\]",
    r"(?i)\[crosstalk\]",
    # Типичные галлюцинации тишины
    r"^\.{3,}$",
    r"^\s*$",
    r"^(?:ага|угу|мм+|эм+|хм+)[,.]?\s*$",
    # TODO: добавьте свои паттерны, найденные в реальных транскрипциях
]


def detect_hallucinations(
    segments: list[dict],
    custom_patterns: list[str] | None = None,
    min_logprob: float = -1.0,
) -> tuple[list[dict], list[dict]]:
    """
    Обнаруживает и убирает галлюцинации из списка сегментов.

    Args:
        segments:        Список сегментов [{text, start, end, avg_logprob, ...}]
        custom_patterns: Дополнительные паттерны regex.
        min_logprob:     Сегменты с avg_logprob < этого значения считаются
                         подозрительными (низкая уверенность Whisper).

    Returns:
        (clean_segments, removed_segments)

    TODO: Реализуйте фильтрацию. Подсказка:
        - Компилируйте паттерны через re.compile() для скорости
        - Проверяйте как text сегмента, так и avg_logprob
        - Записывайте reason: "hallucination_pattern" или "low_logprob"
    """
    all_patterns = HALLUCINATION_PATTERNS + (custom_patterns or [])
    compiled = [re.compile(p, re.IGNORECASE) for p in all_patterns]

    clean, removed = [], []
    for seg in segments:
        text = seg.get("text", "").strip()
        reason = None

        # Проверка по паттернам
        for pattern in compiled:
            if pattern.search(text):
                reason = "hallucination_pattern"
                break

        # Проверка по log-probability
        if reason is None and min_logprob is not None:
            logprob = seg.get("avg_logprob")
            if logprob is not None and logprob < min_logprob:
                reason = "low_logprob"

        if reason:
            removed.append({**seg, "reason": reason})
        else:
            clean.append(seg)

    return clean, removed


def verify_with_vad(
    segments: list[dict],
    speech_intervals: list[dict],
    min_overlap: float = 0.3,
) -> tuple[list[dict], list[dict]]:
    """
    Убирает сегменты, которые слабо перекрываются с речевыми интервалами VAD.

    Логика: если в сегменте нет речи по VAD, но есть текст — вероятно галлюцинация.

    Args:
        segments:         Список текстовых сегментов [{start, end, text}]
        speech_intervals: Речевые интервалы от VAD [{start, end}]
        min_overlap:      Минимальная доля перекрытия (0.0–1.0).

    Returns:
        (clean_segments, removed_segments)

    TODO: Реализуйте функцию.
    Подсказка:
        Для каждого сегмента [seg_start, seg_end] найдите долю перекрытия
        с union всех speech_intervals. Если overlap_ratio < min_overlap — удалить.
    """
    clean, removed = [], []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_dur = max(seg_end - seg_start, 1e-6)

        overlap = 0.0
        for ivl in speech_intervals:
            o = min(seg_end, ivl["end"]) - max(seg_start, ivl["start"])
            if o > 0:
                overlap += o

        overlap_ratio = overlap / seg_dur
        if overlap_ratio < min_overlap:
            removed.append({**seg, "reason": "vad_no_speech",
                            "vad_overlap": round(overlap_ratio, 3)})
        else:
            clean.append(seg)

    return clean, removed


def detect_repetitions(text: str, min_repeat: int = 3, max_ngram: int = 6) -> str:
    """
    Убирает повторяющиеся фразы (looping артефакты Whisper).

    Примеры:
        "да да да да да" → "да"
        "конечно конечно конечно" → "конечно"

    Args:
        text:        Входной текст.
        min_repeat:  Минимальное количество повторений для удаления.
        max_ngram:   Максимальный размер n-граммы для поиска повторений.

    Returns:
        Текст без повторений.
    """
    words = text.split()
    result = list(words)

    # Ищем повторяющиеся n-граммы от длинных к коротким
    for n in range(max_ngram, 0, -1):
        i = 0
        new_result = []
        while i < len(result):
            ngram = tuple(result[i:i+n])
            if len(ngram) < n:
                new_result.extend(result[i:])
                break
            # Считаем сколько раз ngram повторяется подряд
            count = 1
            j = i + n
            while result[j:j+n] and tuple(result[j:j+n]) == ngram:
                count += 1
                j += n
            if count >= min_repeat:
                new_result.extend(list(ngram))  # оставляем одно вхождение
                i = j
            else:
                new_result.append(result[i])
                i += 1
        result = new_result

    return " ".join(result)


def clean_text(text: str) -> str:
    """
    Нормализует текст: убирает лишние пробелы, управляющие символы,
    нормализует Unicode.
    """
    # Нормализация Unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Убираем управляющие символы кроме пробела и переноса строки
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("C")
                   or ch in ("\n", "\t", " "))
    # Множественные пробелы → одиночный
    text = re.sub(r"[ \t]+", " ", text)
    # Множественные переносы строк → два максимум
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
