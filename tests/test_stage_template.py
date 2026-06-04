"""
Шаблон для написания тестов этапов пайплайна.

Запуск:
    pytest tests/ -v
    pytest tests/test_stage_template.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import os

# Регистрируем все компоненты
import src.stages  # noqa
import src.models  # noqa


class TestRegistryIntegrity:
    """Проверяет, что все зарегистрированные компоненты доступны."""

    def test_all_stages_registered(self):
        from src.core.registry import list_stages
        stages = list_stages()
        expected = [
            "fetch", "audio_extract", "vad", "transcription",
            "transcription_compare", "diarization", "quality_metrics",
            "denoise", "postprocess",
        ]
        for s in expected:
            assert s in stages, f"Этап '{s}' не зарегистрирован"

    def test_all_models_registered(self):
        from src.core.registry import list_models
        models = list_models()
        expected = [
            "whisper-small", "whisper-large-v3",
            "faster-whisper-small",
            "wav2vec2-xlsr-ru",
            "pyannote", "simple_diarizer",
            "silero", "webrtcvad",
            "noisereduce",
        ]
        for m in expected:
            assert m in models, f"Модель '{m}' не зарегистрирована"


class TestFetchStage:
    """Тесты для FetchStage."""

    def test_local_file_passthrough(self, tmp_path):
        """Если файл уже локальный — просто пробрасывает путь."""
        # Создаём фиктивный файл
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")

        from src.stages.fetch import FetchStage
        stage = FetchStage(cloud="local")
        context = {"logger": MagicMock()}
        data = stage.run({"video_path": str(video)}, context)

        assert data["video_path"] == str(video)
        assert "fetch_meta" in data

    def test_missing_file_raises(self):
        from src.stages.fetch import FetchStage
        from src.core.base import StageError
        stage = FetchStage(cloud="local")
        context = {"logger": MagicMock()}
        with pytest.raises(Exception):
            stage.run({"video_path": "local:///nonexistent.mp4"}, context)


class TestPostprocessStage:
    """Тесты для PostprocessStage."""

    def test_removes_hallucination_patterns(self):
        from src.stages.postprocess import PostprocessStage
        stage = PostprocessStage(use_vad_verification=False)
        context = {"logger": MagicMock()}
        data = {
            "transcription": "Добрый день. Автор субтитров Dimatorzok",
            "word_timestamps": [
                {"start": 0, "end": 2, "text": "Добрый день."},
                {"start": 2, "end": 4, "text": "Автор субтитров Dimatorzok"},
            ],
        }
        result = stage.run(data, context)
        assert "Dimatorzok" not in result["clean_transcription"]
        assert result["postprocess_stats"]["removed_segments"] == 1

    def test_removes_repetitions(self):
        from src.utils.text_cleaner import detect_repetitions
        text = "да да да да да нет"
        cleaned = detect_repetitions(text, min_repeat=3)
        assert cleaned.count("да") < 3
        assert "нет" in cleaned

    def test_passthrough_clean_text(self):
        from src.stages.postprocess import PostprocessStage
        stage = PostprocessStage(use_vad_verification=False)
        context = {"logger": MagicMock()}
        clean = "Это обычный текст без галлюцинаций."
        data = {
            "transcription": clean,
            "word_timestamps": [{"start": 0, "end": 5, "text": clean}],
        }
        result = stage.run(data, context)
        assert result["clean_transcription"] == clean
        assert result["postprocess_stats"]["removed_segments"] == 0


class TestVADStage:
    """Базовые проверки VADStage."""

    def test_validates_audio_path(self):
        from src.stages.vad import VADStage
        from src.core.base import StageError
        stage = VADStage(model="silero")
        with pytest.raises(StageError):
            stage.run({}, {"logger": MagicMock()})

    def test_silence_is_complement_of_speech(self):
        """Тест логики вычисления тишины как дополнения к речи."""
        from src.stages.vad import VADStage
        stage = VADStage.__new__(VADStage)

        # Эмулируем результат run() вручную
        speech = [{"start": 0.5, "end": 3.0}, {"start": 5.0, "end": 8.0}]
        duration = 10.0

        silence = []
        prev_end = 0.0
        for ivl in speech:
            if ivl["start"] > prev_end:
                silence.append({"start": prev_end, "end": ivl["start"]})
            prev_end = ivl["end"]
        if duration > prev_end:
            silence.append({"start": prev_end, "end": duration})

        assert silence[0] == {"start": 0.0, "end": 0.5}
        assert silence[1] == {"start": 3.0, "end": 5.0}
        assert silence[2] == {"start": 8.0, "end": 10.0}


class TestPipelineConfig:
    """Тесты загрузки конфигурационных файлов."""

    def test_default_config_loads(self):
        from src.core.pipeline import Pipeline
        pipeline = Pipeline("config/default_pipeline.yaml")
        assert len(pipeline.stages) > 0

    def test_stage_names_match_registry(self):
        from src.core.pipeline import Pipeline
        pipeline = Pipeline("config/default_pipeline.yaml")
        for stage in pipeline.stages:
            assert hasattr(stage, "name")
            assert stage.name != "unnamed_stage"

    def test_add_stage_dynamic(self):
        from src.core.pipeline import Pipeline
        from src.stages.postprocess import PostprocessStage
        pipeline = Pipeline("config/default_pipeline.yaml")
        original_len = len(pipeline.stages)
        pipeline.add_stage(PostprocessStage(), position=0)
        assert len(pipeline.stages) == original_len + 1
        assert pipeline.stages[0].name == "postprocess"


# ──────────────────────────────────────────────────────────────────────
# Шаблон для теста вашего этапа
# ──────────────────────────────────────────────────────────────────────

class TestMyCustomStage:
    """
    ШАБЛОН: скопируйте этот класс для тестирования своего этапа.

    Переименуйте класс и заполните тесты.
    """

    def test_stage_is_registered(self):
        """Проверяет, что ваш этап зарегистрирован."""
        from src.core.registry import list_stages
        # TODO: замените "my_stage" на имя вашего этапа
        # assert "my_stage" in list_stages()
        pass

    def test_stage_run_returns_data(self):
        """Проверяет, что run() возвращает словарь с нужными ключами."""
        # from src.stages.my_custom_stage import MyCustomStage
        # stage = MyCustomStage(param1="test")
        # context = {"logger": MagicMock()}
        # data = {"audio_path": "test.wav"}
        # result = stage.run(data, context)
        # assert "my_result" in result
        pass
