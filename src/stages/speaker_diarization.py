"""
Этап 4: Диаризация спикеров + объединение с таймкодами транскрибации.
"""
from __future__ import annotations

from src.core.base import Stage
from src.core.registry import register_stage, load_model


@register_stage("diarization")
class DiarizationStage(Stage):
    """
    Определяет, кто и когда говорил, и привязывает спикеров к словам транскрипции.

    Читает из data:
        audio_path (str)
        word_timestamps (list[dict])  — опционально, из TranscriptionStage

    Добавляет в data:
        speaker_turns (list[dict])        — [{start, end, speaker}]
        speaker_transcription (list[dict])— [{speaker, start, end, text}]
        num_speakers (int)
    """

    def __init__(
        self,
        model: str = "pyannote",
        num_speakers: int | None = None,
        device: str = "cpu",
        hf_token: str | None = None,
    ):
        self.model_name = model
        self.num_speakers = num_speakers
        self.device = device
        self.hf_token = hf_token
        self._model = None

    def _get_model(self):
        if self._model is None:
            kwargs = {"device": self.device}
            if self.hf_token:
                kwargs["hf_token"] = self.hf_token
            self._model = load_model(self.model_name, **kwargs)
        return self._model

    def run(self, data: dict, context: dict) -> dict:
        self.validate_inputs(data, ["audio_path"])
        audio_path = data["audio_path"]

        result = self._get_model().predict(
            audio_path,
            num_speakers=self.num_speakers,
        )
        speaker_turns = result["segments"]

        # Объединяем с транскрибацией если есть таймкоды слов
        word_timestamps = data.get("word_timestamps", [])
        speaker_transcription = []
        if word_timestamps:
            speaker_transcription = self._merge_with_transcription(
                word_timestamps, speaker_turns
            )

        speakers = list({s["speaker"] for s in speaker_turns})
        data["speaker_turns"] = speaker_turns
        data["speaker_transcription"] = speaker_transcription
        data["num_speakers"] = len(speakers)
        data["speaker_ids"] = sorted(speakers)
        return data

    @staticmethod
    def _merge_with_transcription(
        word_timestamps: list[dict],
        speaker_turns: list[dict],
    ) -> list[dict]:
        """
        Объединяет таймкоды слов из ASR с сегментами диаризации.

        Алгоритм: для каждого слова находим сегмент диаризации,
        с которым оно максимально перекрывается.

        TODO: Реализуйте этот метод.

        Args:
            word_timestamps: список сегментов из Whisper:
                [{start, end, text, words: [{word, start, end}]}]
            speaker_turns: список сегментов диаризации:
                [{start, end, speaker}]

        Returns:
            Список реплик: [{speaker, start, end, text}]
        """
        # Собираем все слова с таймкодами
        words = []
        for segment in word_timestamps:
            seg_words = segment.get("words", [])
            if seg_words:
                for w in seg_words:
                    words.append({
                        "word": w.get("word", ""),
                        "start": w.get("start", segment["start"]),
                        "end": w.get("end", segment["end"]),
                    })
            else:
                # Если нет пословных таймкодов — добавляем весь сегмент
                words.append({
                    "word": segment.get("text", ""),
                    "start": segment["start"],
                    "end": segment["end"],
                })

        # --- Ваш код здесь ---
        # Для каждого слова найдите спикера из speaker_turns
        # Подсказка: используйте функцию _find_speaker() ниже
        # Затем объедините последовательные слова одного спикера в реплики

        def _find_speaker(word_start: float, word_end: float) -> str:
            """Находит спикера с максимальным перекрытием."""
            best_speaker = "SPEAKER_UNKNOWN"
            best_overlap = 0.0
            for turn in speaker_turns:
                overlap = min(word_end, turn["end"]) - max(word_start, turn["start"])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]
            return best_speaker

        # Присваиваем спикера каждому слову
        for word in words:
            word["speaker"] = _find_speaker(word["start"], word["end"])

        # Объединяем последовательные слова одного спикера
        result = []
        if not words:
            return result

        current = {
            "speaker": words[0]["speaker"],
            "start": words[0]["start"],
            "end": words[0]["end"],
            "words": [words[0]["word"]],
        }
        for w in words[1:]:
            if w["speaker"] == current["speaker"]:
                current["words"].append(w["word"])
                current["end"] = w["end"]
            else:
                result.append({
                    "speaker": current["speaker"],
                    "start": current["start"],
                    "end": current["end"],
                    "text": " ".join(current["words"]).strip(),
                })
                current = {
                    "speaker": w["speaker"],
                    "start": w["start"],
                    "end": w["end"],
                    "words": [w["word"]],
                }
        result.append({
            "speaker": current["speaker"],
            "start": current["start"],
            "end": current["end"],
            "text": " ".join(current["words"]).strip(),
        })
        return result

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "num_speakers": {"type": ["integer", "null"]},
            },
        }
