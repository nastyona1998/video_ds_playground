"""
Этап 2: Извлечение аудио из видео через ffmpeg.
"""
from __future__ import annotations

from pathlib import Path

from src.core.base import Stage
from src.core.registry import register_stage
from src.utils.ffmpeg_helper import extract_audio, get_media_info


@register_stage("audio_extract")
class AudioExtractStage(Stage):
    """
    Извлекает аудиодорожку из видеофайла и конвертирует её
    в формат, пригодный для ASR (16kHz, моно, WAV).

    Читает из data:
        video_path (str)

    Добавляет в data:
        audio_path (str): путь к извлечённому WAV-файлу
        audio_meta (dict): {duration_sec, sample_rate, channels, format, ...}
    """

    def __init__(
        self,
        output_dir: str = "data/audio",
        sample_rate: int = 16000,
        channels: int = 1,
        audio_format: str = "wav",
    ):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_format = audio_format

    def run(self, data: dict, context: dict) -> dict:
        self.validate_inputs(data, ["video_path"])
        video_path = data["video_path"]

        stem = Path(video_path).stem
        suffix = f"_{self.sample_rate // 1000}k_{'mono' if self.channels == 1 else 'stereo'}"
        audio_filename = f"{stem}{suffix}.{self.audio_format}"
        audio_path = str(self.output_dir / audio_filename)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = extract_audio(
            video_path=video_path,
            output_path=audio_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
            audio_format=self.audio_format,
        )

        meta = get_media_info(audio_path)
        meta["original_video"] = str(Path(video_path).name)

        data["audio_path"] = audio_path
        data["audio_meta"] = meta
        return data

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "sample_rate": {"type": "integer", "enum": [8000, 16000, 22050, 44100]},
                "channels": {"type": "integer", "enum": [1, 2]},
                "audio_format": {"type": "string", "enum": ["wav", "flac", "mp3"]},
            },
        }
