"""
Этап 7: Шумоподавление аудио.
"""
from __future__ import annotations

from pathlib import Path

from src.core.base import Stage
from src.core.registry import register_stage, load_model


@register_stage("denoise")
class DenoiseStage(Stage):
    """
    Применяет шумоподавление к аудиофайлу.

    Читает из data:
        audio_path (str)
        metrics.snr_db (float)     — опционально: пропустить если SNR уже хороший

    Добавляет в data:
        denoised_audio_path (str)
        denoise_model (str)
        denoise_skipped (bool)     — True если шумоподавление не применялось
    """

    def __init__(
        self,
        model: str = "noisereduce",
        output_dir: str = "data/audio",
        snr_threshold: float | None = None,
        adaptive: bool = False,
        **model_kwargs,
    ):
        """
        Args:
            model:          Имя модели: "noisereduce", "speechbrain-sepformer", "demucs"
            snr_threshold:  Если SNR > порога — пропустить шумоподавление.
                            None — всегда применять.
            adaptive:       Если True — выбирать силу шумоподавления по SNR.
        """
        self.model_name = model
        self.output_dir = Path(output_dir)
        self.snr_threshold = snr_threshold
        self.adaptive = adaptive
        self.model_kwargs = model_kwargs
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = load_model(self.model_name, **self.model_kwargs)
        return self._model

    def run(self, data: dict, context: dict) -> dict:
        self.validate_inputs(data, ["audio_path"])
        audio_path = data["audio_path"]
        log = context.get("logger")

        # Пропуск по SNR (адаптивная логика)
        snr = data.get("metrics", {}).get("snr_db")
        if self.snr_threshold is not None and snr is not None:
            if snr > self.snr_threshold:
                log.info(f"    SNR={snr:.1f} дБ > порог {self.snr_threshold} дБ: "
                         f"шумоподавление пропущено")
                data["denoised_audio_path"] = audio_path
                data["denoise_model"] = None
                data["denoise_skipped"] = True
                return data

        # Параметры адаптивного шумоподавления
        kwargs = {}
        if self.adaptive and snr is not None:
            # Мягкое подавление при хорошем SNR, агрессивное при плохом
            kwargs["prop_decrease"] = max(0.3, min(1.0, (20 - snr) / 15))
            log.info(f"    Адаптивный режим: SNR={snr:.1f} дБ → "
                     f"prop_decrease={kwargs['prop_decrease']:.2f}")

        # Генерируем путь к выходному файлу
        stem = Path(audio_path).stem
        out_path = str(self.output_dir / f"{stem}_denoised_{self.model_name}.wav")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        result = self._get_model().predict(
            audio_path,
            output_path=out_path,
            **kwargs,
        )
        denoised_path = result.get("output_path", out_path)

        data["denoised_audio_path"] = denoised_path
        data["denoise_model"] = self.model_name
        data["denoise_skipped"] = False
        return data

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "snr_threshold": {"type": ["number", "null"]},
                "adaptive": {"type": "boolean"},
            },
        }
