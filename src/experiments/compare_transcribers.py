"""
Эксперимент: сравнение нескольких ASR-моделей по WER и скорости.

Использование:
    python src/experiments/compare_transcribers.py \\
        --models whisper-small,whisper-large-v3,wav2vec2-xlsr-ru \\
        --video data/raw_videos/sample.mp4 \\
        --reference data/transcripts/sample_ground_truth.txt \\
        --device cpu

Результат: таблица в консоли + HTML-отчёт в data/reports/
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def run(
    models: list[str],
    video_path: str,
    reference_path: str | None = None,
    device: str = "cpu",
    language: str | None = None,
    output_dir: str = "data/reports",
) -> dict:
    # Импортируем после настройки логов, чтобы torch-логи не мешали
    import src.stages  # noqa — регистрация этапов
    import src.models  # noqa — регистрация моделей
    from src.utils.ffmpeg_helper import extract_audio, get_media_info
    from src.utils.metrics import compute_wer, compare_wer_table
    from src.utils.report_generator import generate_report
    from src.core.registry import load_model

    # 1. Извлекаем аудио
    log.info(f"Извлекаю аудио из: {video_path}")
    audio_path = str(Path(video_path).with_suffix("")) + "_16k_mono.wav"
    audio_path = extract_audio(video_path, audio_path)
    audio_meta = get_media_info(audio_path)
    log.info(f"Длительность: {audio_meta.get('duration_sec', 0):.0f} с")

    # 2. Загружаем эталон (если есть)
    reference = None
    if reference_path and Path(reference_path).exists():
        reference = Path(reference_path).read_text(encoding="utf-8").strip()
        log.info(f"Эталон загружен: {len(reference.split())} слов")

    # 3. Прогоняем каждую модель
    results: dict[str, dict] = {}
    for model_name in models:
        log.info(f"\n{'='*50}")
        log.info(f"Модель: {model_name}")
        try:
            model = load_model(model_name, device=device)
            t0 = time.perf_counter()
            pred = model.predict(audio_path, language=language)
            elapsed = time.perf_counter() - t0

            entry = {
                "text": pred.get("text", ""),
                "language": pred.get("language"),
                "inference_time_sec": round(elapsed, 2),
                "wer": None,
            }
            if reference:
                entry["wer"] = compute_wer(reference, entry["text"])
                log.info(f"  WER: {entry['wer']:.4f}")
            log.info(f"  Время: {elapsed:.1f} с")
            results[model_name] = entry

        except NotImplementedError as e:
            log.warning(f"  ⚠  {model_name}: {e}")
            results[model_name] = {"text": "", "wer": None, "inference_time_sec": 0,
                                   "error": str(e)}
        except Exception as e:
            log.error(f"  ✗  {model_name}: {e}")
            results[model_name] = {"text": "", "wer": None, "inference_time_sec": 0,
                                   "error": str(e)}

    # 4. Вывод таблицы
    print("\n" + "=" * 54)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ ТРАНСКРИБАТОРОВ")
    print(compare_wer_table(results))
    print("=" * 54)

    # 5. Генерация отчёта
    report_data = {
        "video_path": video_path,
        "audio_meta": audio_meta,
        "transcription_results": results,
        "ground_truth": reference,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = generate_report(report_data)
    log.info(f"\n✅ Отчёт сохранён: {report_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Сравнение ASR-моделей")
    parser.add_argument("--models", required=True,
                        help="Список моделей через запятую: whisper-small,wav2vec2-xlsr-ru")
    parser.add_argument("--video", required=True, help="Путь к видеофайлу")
    parser.add_argument("--reference", help="Путь к файлу с эталонной транскрипцией")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--language", default=None, help="Язык (ru, en, ...)")
    parser.add_argument("--output-dir", default="data/reports")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    run(models, args.video, args.reference, args.device, args.language, args.output_dir)


if __name__ == "__main__":
    main()
