"""
Класс Pipeline: управляет последовательностью этапов, кешированием,
логированием и сборкой из YAML-конфигурации.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import yaml

from src.core.base import Stage, StageError
from src.core.registry import get_stage

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Оркестратор пайплайна обработки видео.

    Загружает конфигурацию из YAML, инстанциирует этапы через реестр,
    передаёт данные между ними и кеширует промежуточные результаты.

    Пример использования:
        pipeline = Pipeline("config/default_pipeline.yaml")
        result = pipeline.run("data/raw_videos/sample.mp4")
        print(result["clean_transcription"])
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.stages: list[Stage] = self._instantiate_stages()
        self.context: dict = self._build_context()

    # ------------------------------------------------------------------ #
    #  Публичный API                                                       #
    # ------------------------------------------------------------------ #

    def run(self, video_path: str, skip_stages: list[str] | None = None) -> dict:
        """
        Запускает пайплайн для заданного видеофайла.

        Args:
            video_path:   Локальный путь или облачный URI вида nextcloud://...
            skip_stages:  Имена этапов, которые нужно пропустить (используют кеш).

        Returns:
            Итоговый словарь data со всеми результатами.
        """
        skip_stages = skip_stages or []
        data = {"video_path": str(video_path)}
        pipeline_name = self.config.get("name", "pipeline")
        cache_dir = Path(self.config.get("cache_dir", "data/.cache")) / pipeline_name

        logger.info(f"▶  Запуск пайплайна '{pipeline_name}' для: {video_path}")
        total_start = time.perf_counter()

        for stage in self.stages:
            stage_name = stage.name

            # Проверяем кеш
            cache_file = cache_dir / f"{stage_name}.json"
            if stage_name in skip_stages and cache_file.exists():
                logger.info(f"  ⏭  {stage_name}: загружаем из кеша")
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                data.update(cached)
                continue

            logger.info(f"  ▷  {stage_name} ...")
            t0 = time.perf_counter()

            try:
                data = stage.run(data, self.context)
            except StageError as e:
                logger.error(f"  ✗  {stage_name} завершился с ошибкой: {e}")
                raise
            except Exception as e:
                logger.error(f"  ✗  {stage_name} неожиданная ошибка: {e}")
                raise StageError(f"[{stage_name}] {e}") from e

            elapsed = time.perf_counter() - t0
            logger.info(f"  ✓  {stage_name} ({elapsed:.1f}с)")

            # Сохраняем кеш
            if self.config.get("cache_intermediate", True):
                self._save_cache(cache_dir, stage_name, data)

        total_elapsed = time.perf_counter() - total_start
        logger.info(f"✅  Пайплайн завершён за {total_elapsed:.1f}с")
        return data

    def add_stage(self, stage: Stage, position: int | None = None) -> None:
        """
        Динамически добавляет этап в пайплайн.

        Args:
            stage:    Экземпляр класса Stage.
            position: Позиция (0 = начало). None = конец.
        """
        if position is None:
            self.stages.append(stage)
        else:
            self.stages.insert(position, stage)
        logger.info(f"Добавлен этап '{stage.name}' на позицию {position}")

    def skip_stage(self, stage_name: str) -> None:
        """Удаляет этап из пайплайна (не из реестра)."""
        self.stages = [s for s in self.stages if s.name != stage_name]
        logger.info(f"Этап '{stage_name}' исключён из пайплайна")

    def describe(self) -> str:
        """Возвращает текстовое описание текущего пайплайна."""
        lines = [f"Pipeline: {self.config.get('name', '?')}"]
        for i, stage in enumerate(self.stages, 1):
            lines.append(f"  {i}. {stage.name}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Внутренние методы                                                   #
    # ------------------------------------------------------------------ #

    def _load_config(self, path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("pipeline", cfg)

    def _instantiate_stages(self) -> list[Stage]:
        stages = []
        for stage_cfg in self.config.get("stages", []):
            name = stage_cfg["name"]
            params = stage_cfg.get("params", {})
            StageClass = get_stage(name)
            instance = StageClass(**params)
            stages.append(instance)
        return stages

    def _build_context(self) -> dict:
        return {
            "logger": logger,
            "config": self.config,
            "cache_dir": self.config.get("cache_dir", "data/.cache"),
        }

    @staticmethod
    def _save_cache(cache_dir: Path, stage_name: str, data: dict) -> None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Сохраняем только JSON-сериализуемые значения
            serializable = {}
            for k, v in data.items():
                try:
                    json.dumps(v)
                    serializable[k] = v
                except (TypeError, ValueError):
                    pass  # тензоры и объекты пропускаем
            (cache_dir / f"{stage_name}.json").write_text(
                json.dumps(serializable, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Не удалось сохранить кеш для '{stage_name}': {e}")
