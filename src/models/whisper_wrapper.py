"""
Обёртка для OpenAI Whisper и faster-whisper.

Установка:
    pip install openai-whisper
    pip install faster-whisper   # опционально
"""
from __future__ import annotations

import time
from typing import Any

from src.core.base import ModelWrapper
from src.core.registry import register_model


@register_model("whisper-tiny")
@register_model("whisper-base")
@register_model("whisper-small")
@register_model("whisper-medium")
@register_model("whisper-large")
@register_model("whisper-large-v2")
@register_model("whisper-large-v3")
class WhisperWrapper(ModelWrapper):
    """
    Обёртка для оригинального OpenAI Whisper.

    Поддерживает все размеры: tiny, base, small, medium, large, large-v2, large-v3.
    """

    # Маппинг имён моделей из реестра → размеры Whisper
    SIZE_MAP = {
        "whisper-tiny": "tiny",
        "whisper-base": "base",
        "whisper-small": "small",
        "whisper-medium": "medium",
        "whisper-large": "large",
        "whisper-large-v2": "large-v2",
        "whisper-large-v3": "large-v3",
    }

    def load(self, model_size: str = "small", device: str = "cpu", **kwargs) -> None:
        try:
            import whisper
        except ImportError:
            raise ImportError("Установите: pip install openai-whisper")

        # Определяем размер из имени модели, если передан полный идентификатор
        if model_size.startswith("whisper-"):
            model_size = self.SIZE_MAP.get(model_size, "small")

        self.model = whisper.load_model(model_size, device=device)
        self._model_size = model_size
        self._device = device

    def predict(self, audio_path: str, language: str | None = None,
                word_timestamps: bool = True, **kwargs) -> dict:
        t0 = time.perf_counter()
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result.get("language"),
            "inference_time_sec": round(elapsed, 2),
        }


@register_model("faster-whisper-small")
@register_model("faster-whisper-large-v3")
class FasterWhisperWrapper(ModelWrapper):
    """
    Обёртка для faster-whisper (CTranslate2, в 2–4× быстрее оригинала).

    Установка: pip install faster-whisper
    """

    SIZE_MAP = {
        "faster-whisper-tiny": "tiny",
        "faster-whisper-base": "base",
        "faster-whisper-small": "small",
        "faster-whisper-medium": "medium",
        "faster-whisper-large-v3": "large-v3",
    }

    def load(self, model_size: str = "small", device: str = "cpu",
             compute_type: str = "int8", **kwargs) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("Установите: pip install faster-whisper")

        if model_size.startswith("faster-whisper-"):
            model_size = self.SIZE_MAP.get(model_size, "small")

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def predict(self, audio_path: str, language: str | None = None,
                beam_size: int = 5, **kwargs) -> dict:
        t0 = time.perf_counter()
        segments_gen, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            word_timestamps=True,
        )
        segments = []
        full_text = []
        for seg in segments_gen:
            words = [{"word": w.word, "start": w.start, "end": w.end,
                      "probability": w.probability}
                     for w in (seg.words or [])]
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": words,
                "avg_logprob": seg.avg_logprob,
                "no_speech_prob": seg.no_speech_prob,
            })
            full_text.append(seg.text)

        elapsed = time.perf_counter() - t0
        return {
            "text": " ".join(full_text).strip(),
            "segments": segments,
            "language": info.language,
            "inference_time_sec": round(elapsed, 2),
        }
