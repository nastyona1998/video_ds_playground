"""
Обёртка для Silero VAD.

Установка: pip install silero-vad   # или torch.hub.load
"""
from __future__ import annotations

from src.core.base import ModelWrapper
from src.core.registry import register_model


@register_model("silero")
class SileroVADWrapper(ModelWrapper):
    """
    Быстрый VAD на ONNX/PyTorch от silero-models.

    Задание: docs/tasks/05_vad.md
    """

    def load(self, **kwargs) -> None:
        """
        TODO: Реализуйте загрузку Silero VAD.

        Подсказка:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False
            )
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils
            self.model = model
        """
        raise NotImplementedError(
            "Реализуйте load() для SileroVADWrapper.\n"
            "Смотрите задание: docs/tasks/05_vad.md"
        )

    def predict(
        self,
        audio_path: str,
        threshold: float = 0.5,
        min_silence_ms: int = 100,
        min_speech_ms: int = 250,
        **kwargs,
    ) -> dict:
        """
        TODO: Реализуйте детекцию речи.

        Подсказка:
            wav = self.read_audio(audio_path, sampling_rate=16000)
            timestamps = self.get_speech_timestamps(
                wav, self.model,
                threshold=threshold,
                min_silence_duration_ms=min_silence_ms,
                min_speech_duration_ms=min_speech_ms,
                return_seconds=True
            )
            # timestamps: [{"start": float, "end": float}, ...]
            return {"speech_intervals": timestamps}
        """
        raise NotImplementedError("Реализуйте predict() для SileroVADWrapper.")


@register_model("webrtcvad")
class WebRTCVADWrapper(ModelWrapper):
    """
    Алгоритмический VAD от Google (WebRTC).
    Работает без GPU, очень быстро.

    Установка: pip install webrtcvad
    """

    def load(self, aggressiveness: int = 2, **kwargs) -> None:
        try:
            import webrtcvad
        except ImportError:
            raise ImportError("Установите: pip install webrtcvad")
        self.vad = webrtcvad.Vad(aggressiveness)  # 0=мягкий, 3=агрессивный
        self._aggressiveness = aggressiveness

    def predict(self, audio_path: str, frame_duration_ms: int = 30, **kwargs) -> dict:
        import wave
        import struct

        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        # WebRTC VAD поддерживает только 8000/16000/32000/48000 Гц и моно
        assert sample_rate in (8000, 16000, 32000, 48000), (
            f"WebRTC VAD поддерживает только 8/16/32/48 кГц, получено: {sample_rate}"
        )

        frame_size = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit = 2 bytes
        speech_intervals = []
        is_speech = False
        speech_start = 0.0

        for i in range(0, len(frames) - frame_size, frame_size):
            chunk = frames[i: i + frame_size]
            t = i / (sample_rate * 2)  # время в секундах
            voiced = self.vad.is_speech(chunk, sample_rate)
            if voiced and not is_speech:
                speech_start = t
                is_speech = True
            elif not voiced and is_speech:
                speech_intervals.append({"start": speech_start, "end": t})
                is_speech = False

        if is_speech:
            total_sec = len(frames) / (sample_rate * 2)
            speech_intervals.append({"start": speech_start, "end": total_sec})

        return {"speech_intervals": speech_intervals}
