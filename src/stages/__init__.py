# Импорт всех этапов — необходим для регистрации в STAGE_REGISTRY
from src.stages.fetch import FetchStage
from src.stages.audio_extractor import AudioExtractStage
from src.stages.vad import VADStage
from src.stages.transcription import TranscriptionStage, TranscriptionCompareStage
from src.stages.speaker_diarization import DiarizationStage
from src.stages.quality_metrics import QualityMetricsStage
from src.stages.denoise import DenoiseStage
from src.stages.postprocess import PostprocessStage

__all__ = [
    "FetchStage",
    "AudioExtractStage",
    "VADStage",
    "TranscriptionStage",
    "TranscriptionCompareStage",
    "DiarizationStage",
    "QualityMetricsStage",
    "DenoiseStage",
    "PostprocessStage",
]
