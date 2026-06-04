# Импорт всех моделей — необходим для регистрации в MODEL_REGISTRY
from src.models.whisper_wrapper import WhisperWrapper, FasterWhisperWrapper
from src.models.wav2vec2_wrapper import Wav2Vec2Wrapper
from src.models.pyannote_wrapper import PyannoteWrapper, SimpleDiarizerWrapper
from src.models.silero_vad import SileroVADWrapper, WebRTCVADWrapper
from src.models.denoise_wrappers import NoiseReduceWrapper, SpeechBrainDenoiser, DemucsWrapper

__all__ = [
    "WhisperWrapper",
    "FasterWhisperWrapper",
    "Wav2Vec2Wrapper",
    "PyannoteWrapper",
    "SimpleDiarizerWrapper",
    "SileroVADWrapper",
    "WebRTCVADWrapper",
    "NoiseReduceWrapper",
    "SpeechBrainDenoiser",
    "DemucsWrapper",
]
