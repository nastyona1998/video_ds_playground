from __future__ import annotations
from src.core.base import ModelWrapper
from src.core.registry import register_model


@register_model("silero")
class SileroVADWrapper(ModelWrapper):

    def load(self, **kwargs) -> None:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False, trust_repo=True
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        self.model = model

    def predict(self, audio_path: str, threshold: float = 0.5,
                min_silence_ms: int = 100, min_speech_ms: int = 250, **kwargs) -> dict:
        wav = self.read_audio(audio_path, sampling_rate=16000)
        timestamps = self.get_speech_timestamps(
            wav, self.model,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            min_speech_duration_ms=min_speech_ms,
            return_seconds=True
        )
        return {"speech_intervals": timestamps}


@register_model("webrtcvad")
class WebRTCVADWrapper(ModelWrapper):

    def load(self, aggressiveness: int = 2, **kwargs) -> None:
        try:
            import webrtcvad
        except ImportError:
            raise ImportError("Установите: pip install webrtcvad")
        self.vad = webrtcvad.Vad(aggressiveness)
        self._aggressiveness = aggressiveness

    def predict(self, audio_path: str, frame_duration_ms: int = 30, **kwargs) -> dict:
        import wave

        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        assert sample_rate in (8000, 16000, 32000, 48000)

        frame_size = int(sample_rate * frame_duration_ms / 1000) * 2
        speech_intervals = []
        is_speech = False
        speech_start = 0.0

        for i in range(0, len(frames) - frame_size, frame_size):
            chunk = frames[i: i + frame_size]
            t = i / (sample_rate * 2)
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
