"""
Эксперимент: влияние шумоподавления на WER разных ASR-моделей.

Ключевой вывод: сильные модели (Whisper large) устойчивы к шуму,
слабые модели (wav2vec2) выигрывают от предварительной очистки.

Использование:
    python src/experiments/denoise_impact.py \\
        --denoisers noisereduce,speechbrain-sepformer \\
        --transcribers whisper-small,whisper-large-v3,wav2vec2-xlsr-ru \\
        --video data/raw_videos/noisy_sample.mp4 \\
        --reference data/transcripts/sample_ground_truth.txt
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def run(
    denoisers: list[str],
    transcribers: list[str],
    video_path: str,
    reference_path: str | None = None,
    device: str = "cpu",
    language: str | None = None,
) -> list[dict]:
    import src.stages  # noqa
    import src.models  # noqa
    from src.utils.ffmpeg_helper import extract_audio
    from src.utils.metrics import compute_wer, compute_snr
    from src.utils.report_generator import generate_report
    from src.core.registry import load_model

    # Извлекаем аудио
    audio_path = str(Path(video_path).with_suffix("")) + "_16k_mono.wav"
    audio_path = extract_audio(video_path, audio_path)

    reference = None
    if reference_path and Path(reference_path).exists():
        reference = Path(reference_path).read_text(encoding="utf-8").strip()

    original_snr = compute_snr(audio_path)
    log.info(f"Исходный SNR: {original_snr:.1f} дБ")

    # Таблица результатов: (denoiser, transcriber, audio_path)
    audio_variants: dict[str, str] = {"baseline": audio_path}

    # Шаг 1: применяем каждый метод шумоподавления
    for denoiser_name in denoisers:
        log.info(f"\nПрименяю шумоподавление: {denoiser_name}")
        try:
            denoiser = load_model(denoiser_name)
            out_path = audio_path.replace(".wav", f"_{denoiser_name}.wav")
            result = denoiser.predict(audio_path, output_path=out_path)
            audio_variants[denoiser_name] = result.get("output_path", out_path)
            snr_after = compute_snr(audio_variants[denoiser_name])
            log.info(f"  SNR после: {snr_after:.1f} дБ")
        except NotImplementedError as e:
            log.warning(f"  ⚠  {denoiser_name} не реализован: {e}")
        except Exception as e:
            log.error(f"  ✗  {denoiser_name}: {e}")

    # Шаг 2: транскрибируем каждый вариант каждой моделью
    rows: list[dict] = []
    baseline_wers: dict[str, float] = {}

    for transcriber_name in transcribers:
        log.info(f"\n{'='*50}")
        log.info(f"Транскрибатор: {transcriber_name}")
        try:
            transcriber = load_model(transcriber_name, device=device)
        except NotImplementedError as e:
            log.warning(f"  ⚠  {transcriber_name} не реализован: {e}")
            continue

        for variant_name, variant_audio in audio_variants.items():
            try:
                t0 = time.perf_counter()
                pred = transcriber.predict(variant_audio, language=language)
                elapsed = time.perf_counter() - t0

                wer = compute_wer(reference, pred["text"]) if reference else None
                if variant_name == "baseline":
                    baseline_wers[transcriber_name] = wer

                delta = None
                if wer is not None and transcriber_name in baseline_wers:
                    baseline = baseline_wers[transcriber_name]
                    if baseline is not None and variant_name != "baseline":
                        delta = wer - baseline

                row = {
                    "denoiser": variant_name,
                    "transcriber": transcriber_name,
                    "wer": wer,
                    "delta": delta,
                    "inference_time_sec": round(elapsed, 2),
                }
                rows.append(row)
                wer_str = f"{wer:.4f}" if wer is not None else "N/A"
                delta_str = (f" ({'+' if delta and delta > 0 else ''}{delta:.4f})"
                             if delta is not None else "")
                log.info(f"  {variant_name:<30} WER={wer_str}{delta_str} [{elapsed:.1f}с]")

            except Exception as e:
                log.error(f"  ✗  {variant_name} / {transcriber_name}: {e}")

    # Вывод итоговой таблицы
    print(f"\n{'='*70}")
    print(f"ВЛИЯНИЕ ШУМОПОДАВЛЕНИЯ НА WER  (базовый SNR: {original_snr:.1f} дБ)")
    print(f"{'Шумоподавление':<28} {'Транскрибатор':<25} {'WER':>8} {'ΔWER':>8}")
    print("-" * 70)
    for row in rows:
        delta_str = f"{row['delta']:+.4f}" if row["delta"] is not None else "baseline"
        wer_str = f"{row['wer']:.4f}" if row["wer"] is not None else "N/A"
        print(f"{row['denoiser']:<28} {row['transcriber']:<25} {wer_str:>8} {delta_str:>8}")
    print("=" * 70)

    # Генерируем отчёт
    report_data = {
        "video_path": video_path,
        "denoise_comparison": rows,
        "metrics": {"snr_db": original_snr},
    }
    report_path = generate_report(report_data)
    log.info(f"\n✅ Отчёт сохранён: {report_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Влияние шумоподавления на WER")
    parser.add_argument("--denoisers", default="noisereduce",
                        help="Методы шумоподавления через запятую")
    parser.add_argument("--transcribers", default="whisper-small",
                        help="ASR-модели через запятую")
    parser.add_argument("--video", required=True)
    parser.add_argument("--reference", help="Эталонный текст")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    run(
        denoisers=[d.strip() for d in args.denoisers.split(",")],
        transcribers=[t.strip() for t in args.transcribers.split(",")],
        video_path=args.video,
        reference_path=args.reference,
        device=args.device,
        language=args.language,
    )


if __name__ == "__main__":
    main()
