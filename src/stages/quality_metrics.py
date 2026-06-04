"""
Этап 6: Метрики качества речи (PESQ, STOI, SNR, акцент).
"""
from __future__ import annotations

from pathlib import Path

from src.core.base import Stage
from src.core.registry import register_stage
from src.utils.metrics import compute_pesq, compute_stoi, compute_snr


@register_stage("quality_metrics")
class QualityMetricsStage(Stage):
    """
    Вычисляет инструментальные метрики качества речи.

    Читает из data:
        audio_path (str)
        reference_audio_path (str)   — опционально, для PESQ/STOI

    Добавляет в data:
        metrics (dict): {pesq, stoi, snr_db, duration_sec, speech_ratio}
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        sample_rate: int = 16000,
        reference_key: str = "reference_audio_path",
        accent_model: str | None = None,
    ):
        self.metrics = metrics or ["snr"]
        self.sample_rate = sample_rate
        self.reference_key = reference_key
        self.accent_model = accent_model

    def run(self, data: dict, context: dict) -> dict:
        self.validate_inputs(data, ["audio_path"])
        audio_path = data["audio_path"]
        reference = data.get(self.reference_key)
        result: dict = {}

        if "snr" in self.metrics:
            result["snr_db"] = compute_snr(audio_path)

        if "stoi" in self.metrics:
            if reference:
                result["stoi"] = compute_stoi(reference, audio_path, self.sample_rate)
            else:
                context.get("logger").warning(
                    "STOI требует эталонный сигнал (reference_audio_path). Пропускаем."
                )

        if "pesq" in self.metrics:
            if reference:
                result["pesq"] = compute_pesq(reference, audio_path, self.sample_rate)
            else:
                context.get("logger").warning(
                    "PESQ требует эталонный сигнал (reference_audio_path). Пропускаем."
                )

        if "accent" in self.metrics and self.accent_model:
            from src.core.registry import load_model
            model = load_model(self.accent_model)
            result["accent_scores"] = model.predict(audio_path)

        # Добавляем общую статистику
        result["duration_sec"] = data.get("audio_meta", {}).get("duration_sec")
        result["speech_ratio"] = data.get("speech_ratio")

        data["metrics"] = result
        return data

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["pesq", "stoi", "snr", "accent"]},
                },
            },
        }
