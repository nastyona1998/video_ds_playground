"""
Абстрактные базовые классы для всех компонентов фреймворка.

Студент реализует Stage или ModelWrapper для добавления нового этапа/модели.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Stage(ABC):
    """
    Базовый класс для каждого этапа пайплайна.

    Контракт:
        - run() принимает словарь data и дополняет его результатами этапа.
        - Не изменяет уже существующие ключи (только добавляет новые).
        - При ошибке кидает StageError с описанием проблемы.

    Пример реализации:
        @register_stage("my_stage")
        class MyStage(Stage):
            def run(self, data, context):
                data["result"] = do_something(data["audio_path"])
                return data

            def get_config_schema(self):
                return {"type": "object", "properties": {}}
    """

    # Имя этапа для логирования (переопределяется через @register_stage)
    name: str = "unnamed_stage"

    @abstractmethod
    def run(self, data: dict, context: dict) -> dict:
        """
        Выполняет этап пайплайна.

        Args:
            data:    Словарь с данными, накопленными предыдущими этапами.
            context: Общие ресурсы: logger, cache, pipeline config.

        Returns:
            Обновлённый словарь data с новыми ключами.
        """

    @abstractmethod
    def get_config_schema(self) -> dict:
        """
        Возвращает JSON Schema для параметров этапа.
        Используется для валидации конфигурационного YAML.
        """

    def validate_inputs(self, data: dict, required_keys: list[str]) -> None:
        """Проверяет наличие обязательных ключей в data."""
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise StageError(
                f"[{self.name}] Отсутствуют обязательные ключи: {missing}. "
                f"Доступные ключи: {list(data.keys())}"
            )


class ModelWrapper(ABC):
    """
    Базовый класс для обёрток ML-моделей.

    Отделяет загрузку модели от её использования,
    позволяет легко переключать модели через реестр.

    Пример:
        @register_model("my-model-v1")
        class MyModel(ModelWrapper):
            def load(self, checkpoint="default", device="cpu"):
                self.model = MyLib.from_pretrained(checkpoint)

            def predict(self, audio_path, **params):
                return {"text": self.model.run(audio_path)}
    """

    @abstractmethod
    def load(self, **kwargs) -> None:
        """Загружает модель и её веса. Вызывается один раз при инициализации."""

    @abstractmethod
    def predict(self, audio_path: str, **params) -> Any:
        """
        Выполняет инференс модели.

        Args:
            audio_path: Путь к аудиофайлу (16kHz, моно, WAV).
            **params:   Параметры инференса (language, threshold и т.д.).

        Returns:
            Словарь с результатами. Структура зависит от задачи.
        """

    def is_loaded(self) -> bool:
        return hasattr(self, "model") and self.model is not None


class StageError(Exception):
    """Ошибка при выполнении этапа пайплайна."""
