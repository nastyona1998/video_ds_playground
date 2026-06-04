"""
Этап 8: Постобработка транскрипции.
"""
from __future__ import annotations

from src.core.base import Stage
from src.core.registry import register_stage
from src.utils.text_cleaner import (
    detect_hallucinations,
    detect_repetitions,
    verify_with_vad,
    clean_text,
)


@register_stage("postprocess")
class PostprocessStage(Stage):
    """
    Очищает транскрипцию от галлюцинаций, повторений и артефактов.

    Читает из data:
        transcription (str)
        word_timestamps (list[dict])        — сегменты из ASR
        speech_intervals (list[dict])       — опционально, из VADStage

    Добавляет в data:
        clean_transcription (str)
        removed_fragments (list[dict])      — удалённые фрагменты с причиной
        postprocess_stats (dict)
    """

    def __init__(
        self,
        remove_hallucinations: bool = True,
        remove_repetitions: bool = True,
        use_vad_verification: bool = True,
        min_logprob: float = -1.0,
        min_vad_overlap: float = 0.3,
        custom_patterns: list[str] | None = None,
    ):
        self.remove_hallucinations = remove_hallucinations
        self.remove_repetitions = remove_repetitions
        self.use_vad_verification = use_vad_verification
        self.min_logprob = min_logprob
        self.min_vad_overlap = min_vad_overlap
        self.custom_patterns = custom_patterns or []

    def run(self, data: dict, context: dict) -> dict:
        self.validate_inputs(data, ["transcription"])

        segments = list(data.get("word_timestamps", []))
        speech_intervals = data.get("speech_intervals", [])
        removed = []

        if self.remove_hallucinations and segments:
            segments, removed_h = detect_hallucinations(
                segments,
                custom_patterns=self.custom_patterns,
                min_logprob=self.min_logprob,
            )
            removed.extend(removed_h)

        if self.use_vad_verification and speech_intervals and segments:
            segments, removed_v = verify_with_vad(
                segments,
                speech_intervals,
                min_overlap=self.min_vad_overlap,
            )
            removed.extend(removed_v)

        # Собираем чистый текст из оставшихся сегментов
        if segments:
            clean = " ".join(s.get("text", "") for s in segments).strip()
        else:
            clean = data["transcription"]

        if self.remove_repetitions:
            clean = detect_repetitions(clean)

        clean = clean_text(clean)

        data["clean_transcription"] = clean
        data["removed_fragments"] = removed
        data["postprocess_stats"] = {
            "total_segments": len(data.get("word_timestamps", [])),
            "removed_segments": len(removed),
            "removed_ratio": round(
                len(removed) / max(len(data.get("word_timestamps", [])), 1), 4
            ),
        }
        return data

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "remove_hallucinations": {"type": "boolean"},
                "remove_repetitions": {"type": "boolean"},
                "use_vad_verification": {"type": "boolean"},
                "min_logprob": {"type": "number"},
                "min_vad_overlap": {"type": "number"},
            },
        }
