"""
Этап 5: Детекция голосовой активности (VAD).
"""
from __future__ import annotations

from src.core.base import Stage
from src.core.registry import register_stage, load_model


@register_stage("vad")
class VADStage(Stage):
    """
    Определяет интервалы речи и тишины.

    Читает из data:
        audio_path (str)

    Добавляет в data:
        speech_intervals (list[dict])   — [{start, end}] в секундах
        silence_intervals (list[dict])  — [{start, end}]
        speech_ratio (float)            — доля речи (0.0–1.0)
    """

    def __init__(
        self,
        model: str = "silero",
        threshold: float = 0.5,
        min_silence_ms: int = 100,
        min_speech_ms: int = 250,
        device: str = "cpu",
    ):
        self.model_name = model
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = load_model(self.model_name, device=self.device)
        return self._model

    def run(self, data: dict, context: dict) -> dict:
        self.validate_inputs(data, ["audio_path"])
        audio_path = data["audio_path"]
        duration = data.get("audio_meta", {}).get("duration_sec", 0)

        result = self._get_model().predict(
            audio_path,
            threshold=self.threshold,
            min_silence_ms=self.min_silence_ms,
            min_speech_ms=self.min_speech_ms,
        )
        speech_intervals = result["speech_intervals"]

        # Вычисляем интервалы тишины как дополнение
        silence_intervals = []
        prev_end = 0.0
        for ivl in speech_intervals:
            if ivl["start"] > prev_end:
                silence_intervals.append({"start": prev_end, "end": ivl["start"]})
            prev_end = ivl["end"]
        if duration > prev_end:
            silence_intervals.append({"start": prev_end, "end": duration})

        total_speech = sum(i["end"] - i["start"] for i in speech_intervals)
        speech_ratio = total_speech / duration if duration > 0 else 0.0

        data["speech_intervals"] = speech_intervals
        data["silence_intervals"] = silence_intervals
        data["speech_ratio"] = round(speech_ratio, 4)
        data["total_speech_sec"] = round(total_speech, 2)
        return data

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "threshold": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }
