"""
Генератор HTML-отчётов по результатам пайплайна.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Jinja2-шаблон отчёта (встроен в код — не требует внешних файлов)
REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Video DS Playground — Отчёт</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 2rem; color: #1a1a2e; }
  h1   { color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: .5rem; }
  h2   { color: #0f3460; margin-top: 2rem; }
  h3   { color: #533483; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th   { background: #0f3460; color: white; padding: .6rem 1rem; text-align: left; }
  td   { padding: .5rem 1rem; border-bottom: 1px solid #ddd; }
  tr:hover td { background: #f0f4ff; }
  .best  { background: #d4edda !important; font-weight: bold; }
  .worse { background: #f8d7da !important; }
  .badge { display: inline-block; padding: .2rem .6rem; border-radius: 1rem;
           font-size: .8rem; font-weight: bold; }
  .badge-green  { background: #d4edda; color: #155724; }
  .badge-yellow { background: #fff3cd; color: #856404; }
  .badge-red    { background: #f8d7da; color: #721c24; }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 1rem; margin: 1rem 0; }
  .metric-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
                 padding: 1rem; text-align: center; }
  .metric-val  { font-size: 2rem; font-weight: bold; color: #0f3460; }
  .metric-name { font-size: .85rem; color: #666; }
  .speaker { display: inline-block; padding: .15rem .5rem; border-radius: .3rem;
             font-weight: bold; margin-right: .3rem; font-size: .85rem; }
  .removed { background: #fff3cd; border-left: 4px solid #ffc107; padding: .5rem 1rem;
             margin: .3rem 0; font-size: .9rem; }
  pre  { background: #f8f9fa; padding: 1rem; border-radius: 6px; overflow-x: auto;
         font-size: .9rem; white-space: pre-wrap; }
  .meta { color: #666; font-size: .9rem; }
  .timeline { position: relative; height: 40px; background: #f0f0f0;
              border-radius: 4px; margin: .5rem 0; overflow: hidden; }
  .tl-speech { position: absolute; height: 100%; background: #0f3460; opacity: .7; }
  .footer { text-align: center; color: #aaa; font-size: .85rem; margin-top: 3rem;
            border-top: 1px solid #eee; padding-top: 1rem; }
</style>
</head>
<body>
<h1>📊 Video DS Playground — Отчёт</h1>
<p class="meta">Сформирован: {{ timestamp }} | Файл: {{ filename }}</p>

{% if metrics %}
<h2>Метрики качества аудио</h2>
<div class="metric-grid">
  {% if metrics.snr_db is defined and metrics.snr_db is not none %}
  <div class="metric-card">
    <div class="metric-val">{{ "%.1f"|format(metrics.snr_db) }} дБ</div>
    <div class="metric-name">SNR</div>
  </div>
  {% endif %}
  {% if metrics.pesq is defined and metrics.pesq is not none %}
  <div class="metric-card">
    <div class="metric-val">{{ "%.2f"|format(metrics.pesq) }}</div>
    <div class="metric-name">PESQ (MOS-LQO)</div>
  </div>
  {% endif %}
  {% if metrics.stoi is defined and metrics.stoi is not none %}
  <div class="metric-card">
    <div class="metric-val">{{ "%.3f"|format(metrics.stoi) }}</div>
    <div class="metric-name">STOI</div>
  </div>
  {% endif %}
  {% if metrics.speech_ratio is defined and metrics.speech_ratio is not none %}
  <div class="metric-card">
    <div class="metric-val">{{ "%.0f"|format(metrics.speech_ratio * 100) }}%</div>
    <div class="metric-name">Доля речи</div>
  </div>
  {% endif %}
</div>
{% endif %}

{% if transcription_results %}
<h2>Сравнение ASR-моделей</h2>
<table>
  <tr><th>Модель</th><th>WER</th><th>Время инференса</th><th>Язык</th></tr>
  {% for model, res in transcription_results.items() %}
  <tr class="{{ 'best' if res.wer == best_wer else '' }}">
    <td>{{ model }}</td>
    <td>{% if res.wer is not none %}
      <span class="badge {{ 'badge-green' if res.wer < 0.15 else ('badge-yellow' if res.wer < 0.3 else 'badge-red') }}">
        {{ "%.4f"|format(res.wer) }}
      </span>
    {% else %}—{% endif %}</td>
    <td>{{ "%.1f с"|format(res.inference_time_sec) if res.inference_time_sec else "—" }}</td>
    <td>{{ res.language or "—" }}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}

{% if speech_intervals %}
<h2>VAD — временная шкала речи</h2>
<p class="meta">Речевых сегментов: {{ speech_intervals|length }} | Доля речи: {{ "%.1f"|format(speech_ratio * 100) }}%</p>
<div class="timeline">
  {% for ivl in speech_intervals %}
  <div class="tl-speech" style="left:{{ (ivl.start / duration * 100)|round(2) }}%; width:{{ ((ivl.end - ivl.start) / duration * 100)|round(2) }}%"></div>
  {% endfor %}
</div>
{% endif %}

{% if speaker_transcription %}
<h2>Диаризация спикеров</h2>
{% set speaker_colors = ['#0f3460','#e94560','#533483','#06a77d','#d62246'] %}
{% for turn in speaker_transcription %}
<p>
  <span class="speaker" style="background:{{ speaker_colors[loop.index0 % speaker_colors|length] }}; color:white">
    {{ turn.speaker }}
  </span>
  <span class="meta">[{{ "%.1f"|format(turn.start) }}s – {{ "%.1f"|format(turn.end) }}s]</span>
  {{ turn.text }}
</p>
{% endfor %}
{% endif %}

{% if clean_transcription %}
<h2>Транскрипция</h2>
{% if removed_fragments %}
<h3>Удалённые фрагменты ({{ removed_fragments|length }})</h3>
{% for frag in removed_fragments %}
<div class="removed">
  <strong>{{ frag.reason }}</strong>
  {% if frag.start %} [{{ "%.1f"|format(frag.start) }}s]{% endif %}:
  {{ frag.text }}
</div>
{% endfor %}
{% endif %}
<h3>Очищенный текст</h3>
<pre>{{ clean_transcription }}</pre>
{% endif %}

{% if denoise_comparison %}
<h2>Влияние шумоподавления на WER</h2>
<table>
  <tr><th>Метод шумоподавления</th><th>Транскрибатор</th><th>WER</th><th>ΔWER</th></tr>
  {% for row in denoise_comparison %}
  <tr class="{{ 'best' if row.delta is not none and row.delta < 0 else ('worse' if row.delta is not none and row.delta > 0.02 else '') }}">
    <td>{{ row.denoiser }}</td>
    <td>{{ row.transcriber }}</td>
    <td>{{ "%.4f"|format(row.wer) }}</td>
    <td>{% if row.delta %}{{ "%+.4f"|format(row.delta) }}{% else %}baseline{% endif %}</td>
  </tr>
  {% endfor %}
</table>
{% endif %}

<div class="footer">Video DS Playground · MIT License</div>
</body>
</html>"""


def generate_report(data: dict, output_path: str | None = None) -> str:
    """
    Генерирует HTML-отчёт из словаря результатов пайплайна.

    Args:
        data:        Словарь data после выполнения пайплайна.
        output_path: Путь для сохранения HTML. Если None — сохраняет в data/reports/.

    Returns:
        Путь к созданному HTML-файлу.
    """
    try:
        from jinja2 import Template
    except ImportError:
        raise ImportError("Установите: pip install jinja2")

    # Собираем контекст для шаблона
    filename = Path(data.get("video_path", "unknown")).name
    metrics = data.get("metrics", {})
    transcription_results = data.get("transcription_results", {})

    best_wer = None
    if transcription_results:
        wers = {m: r.get("wer") for m, r in transcription_results.items()
                if r.get("wer") is not None}
        if wers:
            best_wer = min(wers.values())

    speech_intervals = data.get("speech_intervals", [])
    duration = data.get("audio_meta", {}).get("duration_sec", 1) or 1

    ctx = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "filename": filename,
        "metrics": metrics if metrics else None,
        "transcription_results": transcription_results if transcription_results else None,
        "best_wer": best_wer,
        "speech_intervals": speech_intervals if speech_intervals else None,
        "speech_ratio": data.get("speech_ratio", 0),
        "duration": duration,
        "speaker_transcription": data.get("speaker_transcription") or None,
        "clean_transcription": data.get("clean_transcription") or data.get("transcription"),
        "removed_fragments": data.get("removed_fragments") or None,
        "denoise_comparison": data.get("denoise_comparison") or None,
    }

    template = Template(REPORT_TEMPLATE)
    html = template.render(**ctx)

    if output_path is None:
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(reports_dir / f"report_{ts}.html")

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
