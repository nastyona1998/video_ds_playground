"""
Ablation study: оценка вклада каждого этапа пайплайна на итоговый WER.

Запускает N конфигураций, поочерёдно включая/отключая этапы:
    - без VAD vs с VAD
    - без шумоподавления vs с шумоподавлением
    - без постобработки vs с постобработкой

Использование:
    python src/experiments/run_ablation.py \\
        --video data/raw_videos/sample.mp4 \\
        --reference data/transcripts/sample_ground_truth.txt \\
        --transcriber whisper-small
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ABLATION_CONFIGS = [
    {
        "name": "baseline",
        "description": "Только извлечение аудио + транскрибация",
        "stages": ["audio_extract", "transcription"],
    },
    {
        "name": "+ VAD (только речевые сегменты)",
        "description": "VAD фильтрует тишину перед транскрибацией",
        "stages": ["audio_extract", "vad", "transcription"],
    },
    {
        "name": "+ шумоподавление (noisereduce)",
        "description": "Шумоподавление перед транскрибацией",
        "stages": ["audio_extract", "denoise", "transcription"],
    },
    {
        "name": "+ VAD + шумоподавление",
        "description": "Комбинация VAD и шумоподавления",
        "stages": ["audio_extract", "vad", "denoise", "transcription"],
    },
    {
        "name": "+ постобработка",
        "description": "Транскрибация + очистка галлюцинаций",
        "stages": ["audio_extract", "vad", "transcription", "postprocess"],
    },
    {
        "name": "полный пайплайн",
        "description": "Все этапы включены",
        "stages": ["audio_extract", "vad", "denoise", "transcription", "postprocess"],
    },
]


def run(video_path: str, reference_path: str | None, transcriber: str = "whisper-small",
        device: str = "cpu", language: str | None = None) -> list[dict]:
    import src.stages  # noqa
    import src.models  # noqa
    from src.core.pipeline import Pipeline
    from src.utils.metrics import compute_wer
    from src.utils.report_generator import generate_report
    import yaml, tempfile, os

    reference = None
    if reference_path and Path(reference_path).exists():
        reference = Path(reference_path).read_text(encoding="utf-8").strip()

    results = []
    for cfg in ABLATION_CONFIGS:
        log.info(f"\n{'='*55}")
        log.info(f"Конфигурация: {cfg['name']}")

        # Динамически строим YAML-конфиг
        stages_cfg = []
        for stage_name in cfg["stages"]:
            entry: dict = {"name": stage_name, "params": {}}
            if stage_name == "transcription":
                entry["params"] = {"model": transcriber, "device": device,
                                   "language": language}
                if "denoise" in cfg["stages"]:
                    entry["params"]["use_denoised"] = True
            elif stage_name == "denoise":
                entry["params"] = {"model": "noisereduce"}
            elif stage_name == "vad":
                entry["params"] = {"model": "silero"}
            stages_cfg.append(entry)

        pipeline_dict = {
            "pipeline": {
                "name": cfg["name"],
                "cache_intermediate": False,
                "stages": stages_cfg,
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(pipeline_dict, f, allow_unicode=True)
            tmp_path = f.name

        try:
            pipeline = Pipeline(tmp_path)
            data = pipeline.run(video_path)
            text = data.get("clean_transcription") or data.get("transcription", "")
            wer = compute_wer(reference, text) if reference else None

            row = {"config": cfg["name"], "wer": wer, "stages": cfg["stages"]}
            results.append(row)
            log.info(f"WER: {wer:.4f}" if wer is not None else "WER: N/A")
        except Exception as e:
            log.error(f"Ошибка: {e}")
            results.append({"config": cfg["name"], "wer": None, "error": str(e)})
        finally:
            os.unlink(tmp_path)

    # Таблица
    print(f"\n{'='*60}")
    print("ABLATION STUDY")
    print(f"{'Конфигурация':<40} {'WER':>8}")
    print("-" * 50)
    for row in results:
        wer_s = f"{row['wer']:.4f}" if row.get("wer") is not None else "ОШИБКА"
        print(f"{row['config']:<40} {wer_s:>8}")
    print("=" * 60)

    generate_report({"video_path": video_path, "ablation_results": results})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--transcriber", default="whisper-small")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    run(args.video, args.reference, args.transcriber, args.device, args.language)


if __name__ == "__main__":
    main()
