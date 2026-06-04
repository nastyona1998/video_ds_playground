"""
Этап 3: Транскрибация аудио в текст.
"""
from __future__ import annotations

from src.core.base import Stage
from src.core.registry import register_stage, load_model


@register_stage("transcription")
class TranscriptionStage(Stage):
    """
    Транскрибирует аудиофайл с помощью выбранной ASR-модели.

    Читает из data:
        audio_path (str)            — основной аудиофайл
        denoised_audio_path (str)   — опционально, если был этап шумоподавления

    Добавляет в data:
        transcription (str)              — полный текст
        word_timestamps (list[dict])     — [{word, start, end, probability}]
        transcription_language (str)     — определённый язык
        transcription_model (str)        — имя использованной модели
    """

    def __init__(
        self,
        model: str = "whisper-small",
        language: str | None = None,
        use_denoised: bool = False,
        device: str = "cpu",
        **model_kwargs,
    ):
        self.model_name = model
        self.language = language
        self.use_denoised = use_denoised
        self.device = device
        self.model_kwargs = model_kwargs
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = load_model(
                self.model_name, device=self.device, **self.model_kwargs
            )
        return self._model

    def run(self, data: dict, context: dict) -> dict:
        self.validate_inputs(data, ["audio_path"])

        # Выбираем источник аудио
        if self.use_denoised and "denoised_audio_path" in data:
            audio_path = data["denoised_audio_path"]
        else:
            audio_path = data["audio_path"]

        result = self._get_model().predict(
            audio_path,
            language=self.language,
        )

        data["transcription"] = result.get("text", "")
        data["word_timestamps"] = result.get("segments", [])
        data["transcription_language"] = result.get("language", self.language)
        data["transcription_model"] = self.model_name
        return data

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "language": {"type": ["string", "null"]},
                "use_denoised": {"type": "boolean"},
                "device": {"type": "string", "enum": ["cpu", "cuda", "mps"]},
            },
        }


@register_stage("transcription_compare")
class TranscriptionCompareStage(Stage):
    """
    Запускает несколько ASR-моделей и сохраняет результаты для сравнения.

    Добавляет в data:
        transcription_results (dict): {model_name: {text, segments, wer}}
    """

    def __init__(self, models: list[str] | None = None, language: str | None = None,
                 device: str = "cpu", reference_key: str = "ground_truth"):
        self.models = models or ["whisper-small", "whisper-large-v3"]
        self.language = language
        self.device = device
        self.reference_key = reference_key

    def run(self, data: dict, context: dict) -> dict:
        from src.utils.metrics import compute_wer

        self.validate_inputs(data, ["audio_path"])
        audio_path = data["audio_path"]
        reference = data.get(self.reference_key)

        results = {}
        for model_name in self.models:
            model = load_model(model_name, device=self.device)
            result = model.predict(audio_path, language=self.language)
            entry = {
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language"),
            }
            if reference:
                entry["wer"] = compute_wer(reference, entry["text"])
            results[model_name] = entry

        data["transcription_results"] = results
        # Основная транскрипция — лучшая модель по WER или первая в списке
        if reference and results:
            best = min(results, key=lambda m: results[m].get("wer", 999))
        else:
            best = self.models[0]
        data["transcription"] = results[best]["text"]
        data["word_timestamps"] = results[best]["segments"]
        data["transcription_model"] = best
        return data

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "models": {"type": "array", "items": {"type": "string"}},
                "language": {"type": ["string", "null"]},
            },
        }
