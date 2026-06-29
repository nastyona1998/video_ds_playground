"""
Эксперимент: сравнение VAD-моделей по количеству сегментов, доле речи и скорости.

Использование:
    python src/experiments/compare_vad.py \
        --models silero,webrtcvad \
        --video data/raw_videos/trailer.mp4
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def run(models, video_path, output_dir="data/reports"):
    import src.stages  # noqa
    import src.models  # noqa
    from src.utils.ffmpeg_helper import extract_audio, get_media_info
    from src.core.registry import load_model

    log.info(f"Извлекаю аудио из: {video_path}")
    audio_path = str(Path(video_path).with_suffix("")) + "_16k_mono.wav"
    audio_path = extract_audio(video_path, audio_path)
    audio_meta = get_media_info(audio_path)
    duration = audio_meta.get("duration_sec", 1) or 1
    log.info(f"Длительность: {duration:.0f} с")

    results = {}
    for model_name in models:
        log.info(f"\n{'='*50}")
        log.info(f"VAD модель: {model_name}")
        try:
            model = load_model(model_name)
            t0 = time.perf_counter()
            pred = model.predict(audio_path)
            elapsed = time.perf_counter() - t0
            intervals = pred.get("speech_intervals", [])
            speech_duration = sum(i["end"] - i["start"] for i in intervals)
            speech_ratio = speech_duration / duration
            results[model_name] = {
                "num_segments": len(intervals),
                "speech_duration_sec": round(speech_duration, 2),
                "speech_ratio": round(speech_ratio, 4),
                "inference_time_sec": round(elapsed, 2),
                "intervals": intervals,
            }
            log.info(f"  Сегментов: {len(intervals)}")
            log.info(f"  Доля речи: {speech_ratio*100:.1f}%")
            log.info(f"  Время: {elapsed:.1f} с")
        except Exception as e:
            log.error(f"  ✗  {model_name}: {e}")
            results[model_name] = {"error": str(e), "num_segments": 0,
                                   "speech_ratio": 0, "inference_time_sec": 0}

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ VAD")
    print(f"{'Модель':<20} {'Сегментов':>10} {'Доля речи':>12} {'Время (с)':>12}")
    print("-" * 60)
    for m, d in results.items():
        if "error" in d:
            print(f"{m:<20} {'ОШИБКА':>10}")
        else:
            print(f"{m:<20} {d['num_segments']:>10} "
                  f"{d['speech_ratio']*100:>11.1f}% "
                  f"{d['inference_time_sec']:>12.1f}")
    print("=" * 60)

    # HTML отчёт
    rows = ""
    for m, d in results.items():
        if "error" in d:
            rows += f"<tr><td>{m}</td><td colspan=3>Ошибка: {d['error']}</td></tr>"
        else:
            rows += (f"<tr><td>{m}</td><td>{d['num_segments']}</td>"
                     f"<td>{d['speech_ratio']*100:.1f}%</td>"
                     f"<td>{d['inference_time_sec']:.1f} с</td></tr>")

    timelines = ""
    for m, d in results.items():
        if "intervals" not in d:
            continue
        bars = ""
        for ivl in d["intervals"]:
            left = round(ivl["start"] / duration * 100, 2)
            width = round((ivl["end"] - ivl["start"]) / duration * 100, 2)
            bars += (f'<div style="position:absolute;height:100%;background:#0f3460;'
                     f'opacity:.7;left:{left}%;width:{width}%"></div>')
        timelines += (f"<h3>{m} — {d['num_segments']} сегментов, "
                      f"{d['speech_ratio']*100:.1f}% речи</h3>"
                      f'<div style="position:relative;height:40px;background:#f0f0f0;'
                      f'border-radius:4px;margin:.5rem 0;overflow:hidden">{bars}</div>')

    css = ("body{font-family:sans-serif;max-width:1100px;margin:0 auto;padding:2rem}"
           "h1{color:#16213e;border-bottom:3px solid #0f3460;padding-bottom:.5rem}"
           "h2,h3{color:#0f3460}"
           "table{border-collapse:collapse;width:100%;margin:1rem 0}"
           "th{background:#0f3460;color:white;padding:.6rem 1rem;text-align:left}"
           "td{padding:.5rem 1rem;border-bottom:1px solid #ddd}")

    html = (f'<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">'
            f'<title>VAD сравнение</title><style>{css}</style></head><body>'
            f'<h1>Сравнение VAD-моделей</h1>'
            f'<p>Файл: {Path(video_path).name} | Длительность: {duration:.0f} с | '
            f'{datetime.now().strftime("%Y-%m-%d %H:%M")}</p>'
            f'<h2>Таблица результатов</h2>'
            f'<table><tr><th>Модель</th><th>Сегментов</th><th>Доля речи</th>'
            f'<th>Время</th></tr>{rows}</table>'
            f'<h2>Временные шкалы речи</h2>{timelines}</body></html>')

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = str(Path(output_dir) / f"vad_compare_{ts}.html")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(html, encoding="utf-8")
    log.info(f"\n✅ Отчёт сохранён: {report_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Сравнение VAD-моделей")
    parser.add_argument("--models", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="data/reports")
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",")]
    run(models, args.video, args.output_dir)


if __name__ == "__main__":
    main()
