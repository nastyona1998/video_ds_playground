"""
Реестр этапов и моделей.

Позволяет добавлять новые компоненты без изменения ядра фреймворка.
Используйте декораторы @register_stage и @register_model.

Пример:
    @register_stage("my_stage")
    class MyStage(Stage): ...

    @register_model("my-model-v1")
    class MyModel(ModelWrapper): ...

    # Получить класс по имени:
    StageClass = get_stage("my_stage")
    ModelClass = get_model("my-model-v1")
"""
from __future__ import annotations

from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.base import Stage, ModelWrapper

# Глобальные реестры
STAGE_REGISTRY: dict[str, Type["Stage"]] = {}
MODEL_REGISTRY: dict[str, Type["ModelWrapper"]] = {}


def register_stage(name: str):
    """
    Декоратор для регистрации класса этапа по строковому имени.

    Использование:
        @register_stage("transcription")
        class TranscriptionStage(Stage): ...
    """
    def decorator(cls):
        if name in STAGE_REGISTRY:
            raise ValueError(
                f"Этап '{name}' уже зарегистрирован. "
                f"Существующий класс: {STAGE_REGISTRY[name].__name__}"
            )
        cls.name = name
        STAGE_REGISTRY[name] = cls
        return cls
    return decorator


def register_model(name: str):
    """
    Декоратор для регистрации обёртки модели по строковому имени.

    Один класс может быть зарегистрирован под несколькими именами:
        @register_model("whisper-small")
        @register_model("whisper-base")
        class WhisperWrapper(ModelWrapper): ...
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_stage(name: str) -> Type["Stage"]:
    """Возвращает класс этапа по имени. Кидает KeyError если не найден."""
    if name not in STAGE_REGISTRY:
        available = list(STAGE_REGISTRY.keys())
        raise KeyError(
            f"Этап '{name}' не найден в реестре.\n"
            f"Доступные этапы: {available}\n"
            f"Убедитесь, что файл с классом импортирован в src/stages/__init__.py"
        )
    return STAGE_REGISTRY[name]


def get_model(name: str) -> Type["ModelWrapper"]:
    """Возвращает класс модели по имени. Кидает KeyError если не найден."""
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Модель '{name}' не найдена в реестре.\n"
            f"Доступные модели: {available}\n"
            f"Убедитесь, что файл с классом импортирован в src/models/__init__.py"
        )
    return MODEL_REGISTRY[name]


def list_stages() -> list[str]:
    return sorted(STAGE_REGISTRY.keys())


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def load_model(name: str, **kwargs) -> "ModelWrapper":
    """Создаёт экземпляр модели и вызывает load()."""
    ModelClass = get_model(name)
    instance = ModelClass()
    instance.load(**kwargs)
    return instance
