"""
Обёртка для pyannote.audio (диаризация спикеров).

Требует HuggingFace токена и принятия условий использования:
    https://huggingface.co/pyannote/speaker-diarization-3.1

Установка: pip install pyannote.audio
"""
from __future__ import annotations

import os

from src.core.base import ModelWrapper
from src.core.registry import register_model


@register_model("pyannote")
class PyannoteWrapper(ModelWrapper):
    """
    Диаризация спикеров через pyannote.audio 3.x.

    Задание: docs/tasks/04_speaker_diarization.md
    """

    def load(
        self,
        hf_token: str | None = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        from pyannote.audio import Pipeline
        import torch
        token = hf_token or os.environ.get("HF_TOKEN")
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=token
        )
        self.pipeline.to(torch.device(device))


    def predict(
        self,
        audio_path: str,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        **kwargs,
    ) -> dict:
        params = {}
        if num_speakers:
            params["num_speakers"] = num_speakers
        elif min_speakers or max_speakers:
            params["min_speakers"] = min_speakers
            params["max_speakers"] = max_speakers

        diarization = self.pipeline(audio_path, **params)
        segments = []
        annotation = diarization.speaker_diarization if hasattr(diarization, 'speaker_diarization') else diarization
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        return {"segments": segments}


@register_model("simple_diarizer")
class SimpleDiarizerWrapper(ModelWrapper):
    """
    Упрощённая диаризация без HuggingFace токена.

    Установка: pip install simple_diarizer
    """

    def load(self, embed_model: str = "xvec", **kwargs) -> None:
        try:
            from simple_diarizer.diarizer import Diarizer
        except ImportError:
            raise ImportError("Установите: pip install simple_diarizer")
        self.model = Diarizer(
            embed_model=embed_model,
            cluster_method="sc",
        )

    def predict(self, audio_path: str, num_speakers: int | None = None, **kwargs) -> dict:
        segments = self.model.diarize(
            audio_path,
            num_speakers=num_speakers,
        )
        result = []
        for seg in segments:
            result.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": f"SPEAKER_{seg['label']:02d}",
            })
        return {"segments": result}
