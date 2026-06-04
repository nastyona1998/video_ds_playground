"""
Метрики качества речи и транскрибации.

Задание (docs/tasks/06_quality_metrics.md):
    Реализуйте функции compute_pesq(), compute_stoi(), compute_snr().
"""
from __future__ import annotations

import numpy as np


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Вычисляет Word Error Rate (WER).

    WER = (S + D + I) / N
        S — замены, D — удаления, I — вставки, N — слов в эталоне.

    Args:
        reference:  Эталонный текст.
        hypothesis: Текст от модели.

    Returns:
        WER в диапазоне 0.0 ... (может быть > 1.0 при многих вставках).
    """
    try:
        from jiwer import wer
        return wer(reference, hypothesis)
    except ImportError:
        # Простая реализация на случай отсутствия jiwer
        return _wer_naive(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Вычисляет Character Error Rate (CER)."""
    try:
        from jiwer import cer
        return cer(reference, hypothesis)
    except ImportError:
        return _wer_naive(
            " ".join(list(reference.replace(" ", ""))),
            " ".join(list(hypothesis.replace(" ", "")))
        )


def compute_pesq(
    reference_path: str,
    degraded_path: str,
    sample_rate: int = 16000,
    mode: str = "wb",
) -> float:
    """
    Вычисляет PESQ (ITU-T P.862).

    Диапазон MOS-LQO: -0.5 ... 4.5 (выше = лучше).
    mode: "wb" = wideband (16 кГц), "nb" = narrowband (8 кГц).

    Требует эталонный сигнал (clean reference).

    TODO: Реализуйте эту функцию.

    Подсказка:
        import torchaudio
        import torchaudio.functional as F

        ref, sr_ref = torchaudio.load(reference_path)
        deg, sr_deg = torchaudio.load(degraded_path)

        # Убедитесь что оба файла одинаковой длины
        min_len = min(ref.shape[-1], deg.shape[-1])
        ref, deg = ref[..., :min_len], deg[..., :min_len]

        pesq_score = F.pesq(sample_rate, ref, deg, mode)
        return float(pesq_score.mean())
    """
    raise NotImplementedError(
        "Реализуйте compute_pesq().\n"
        "Смотрите задание: docs/tasks/06_quality_metrics.md"
    )


def compute_stoi(
    reference_path: str,
    degraded_path: str,
    sample_rate: int = 16000,
    extended: bool = False,
) -> float:
    """
    Вычисляет STOI (Short-Time Objective Intelligibility).

    Диапазон: 0.0 ... 1.0 (выше = лучше разборчивость).

    TODO: Реализуйте эту функцию.

    Подсказка:
        import torchaudio
        import torchaudio.functional as F

        ref, _ = torchaudio.load(reference_path)
        deg, _ = torchaudio.load(degraded_path)
        min_len = min(ref.shape[-1], deg.shape[-1])
        ref, deg = ref[..., :min_len], deg[..., :min_len]

        stoi_score = F.stoi(ref, deg, sample_rate, extended=extended)
        return float(stoi_score.mean())
    """
    raise NotImplementedError(
        "Реализуйте compute_stoi().\n"
        "Смотрите задание: docs/tasks/06_quality_metrics.md"
    )


def compute_snr(audio_path: str, use_vad: bool = True) -> float:
    """
    Оценивает SNR (Signal-to-Noise Ratio) в дБ без эталонного сигнала.

    Алгоритм: делит запись на речевые (сигнал) и тихие (шум) сегменты
    с помощью простого порога энергии, затем вычисляет отношение.

    Args:
        audio_path: путь к WAV-файлу.
        use_vad:    использовать порог энергии для разделения сигнал/шум.

    Returns:
        SNR в дБ. Типичные значения: < 10 = шумно, > 20 = чисто.

    TODO: Для более точной оценки подключите Silero VAD вместо порога энергии.
    """
    import soundfile as sf

    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)

    if not use_vad:
        # Грубая оценка: первые 0.5 сек считаем шумом
        noise_len = int(sr * 0.5)
        noise = data[:noise_len]
        signal = data[noise_len:]
    else:
        # Порог по RMS: фреймы ниже 20% максимума считаем шумом
        frame_size = int(sr * 0.02)  # 20 мс
        frames = [data[i:i+frame_size] for i in range(0, len(data)-frame_size, frame_size)]
        rms_values = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        threshold = 0.2 * rms_values.max()
        signal_frames = [f for f, r in zip(frames, rms_values) if r >= threshold]
        noise_frames = [f for f, r in zip(frames, rms_values) if r < threshold]
        signal = np.concatenate(signal_frames) if signal_frames else data
        noise = np.concatenate(noise_frames) if noise_frames else data[:int(sr * 0.5)]

    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return float("inf")

    snr_db = 10 * np.log10(signal_power / noise_power)
    return round(float(snr_db), 2)


# ------------------------------------------------------------------ #
#  Вспомогательные функции                                            #
# ------------------------------------------------------------------ #

def _wer_naive(reference: str, hypothesis: str) -> float:
    """Наивная реализация WER через расстояние Левенштейна."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    n = len(ref_words)
    if n == 0:
        return 0.0 if len(hyp_words) == 0 else float("inf")

    # Матрица редакционных расстояний
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref_words)][len(hyp_words)] / n


def compare_wer_table(results: dict[str, dict]) -> str:
    """
    Форматирует таблицу сравнения WER для нескольких моделей.

    Args:
        results: {model_name: {"wer": float, "inference_time_sec": float}}

    Returns:
        Строка с ASCII-таблицей.
    """
    lines = [
        f"{'Модель':<30} {'WER':>8} {'Время (с)':>12}",
        "-" * 54,
    ]
    for model, data in sorted(results.items(), key=lambda x: x[1].get("wer", 999)):
        wer_val = data.get("wer", float("nan"))
        time_val = data.get("inference_time_sec", float("nan"))
        lines.append(f"{model:<30} {wer_val:>8.4f} {time_val:>12.1f}")
    return "\n".join(lines)
