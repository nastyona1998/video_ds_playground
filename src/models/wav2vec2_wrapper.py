from __future__ import annotations
import time
from src.core.base import ModelWrapper
from src.core.registry import register_model


@register_model("wav2vec2-xlsr-ru")
@register_model("wav2vec2-ru-golos")
class Wav2Vec2Wrapper(ModelWrapper):

    MODEL_NAMES = {
        "wav2vec2-xlsr-ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        "wav2vec2-ru-golos": "bond005/wav2vec2-large-ru-golos",
    }

    def load(self, model_name: str = "wav2vec2-xlsr-ru", device: str = "cpu", **kwargs) -> None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import torch

        hf_name = self.MODEL_NAMES.get(model_name, model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(hf_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(hf_name)
        self.model.to(device)
        self.model.eval()
        self._device = device

    def predict(self, audio_path: str, **kwargs) -> dict:
        import torchaudio
        import torch

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        with torch.no_grad():
            logits = self.model(**inputs.to(self._device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return {"text": transcription, "segments": [], "language": "ru"}
