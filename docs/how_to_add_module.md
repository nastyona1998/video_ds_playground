# Как добавить новый этап или модель

## Добавление нового этапа

### 1. Создайте файл в `src/stages/`

```python
# src/stages/my_custom_stage.py
from src.core.base import Stage
from src.core.registry import register_stage

@register_stage("my_stage")
class MyCustomStage(Stage):
    """
    Описание: что делает ваш этап.
    
    Ожидает в data:
        - audio_path (str): путь к аудиофайлу
    
    Добавляет в data:
        - my_result (dict): результат обработки
    """

    def __init__(self, param1: str = "default", param2: float = 0.5):
        self.param1 = param1
        self.param2 = param2

    def run(self, data: dict, context: dict) -> dict:
        audio_path = data["audio_path"]
        
        # Ваша логика
        result = self._process(audio_path)
        
        data["my_result"] = result
        return data

    def _process(self, audio_path: str) -> dict:
        # Реализация
        return {}

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
```

### 2. Импортируйте в `src/stages/__init__.py`

```python
from src.stages.my_custom_stage import MyCustomStage
```

### 3. Добавьте этап в конфиг YAML

```yaml
stages:
  - name: my_stage
    class: MyCustomStage
    params:
      param1: "hello"
      param2: 0.7
```

---

## Добавление новой модели

### 1. Создайте обёртку в `src/models/`

```python
# src/models/my_model_wrapper.py
from src.core.base import ModelWrapper
from src.core.registry import register_model

@register_model("my-model-v1")
class MyModelWrapper(ModelWrapper):
    
    def load(self, checkpoint: str = "default", device: str = "cpu"):
        # Загрузка модели
        import my_library
        self.model = my_library.load(checkpoint)
        self.model.to(device)

    def predict(self, audio_path: str, **params) -> dict:
        # Инференс
        result = self.model.run(audio_path)
        return {
            "text": result.text,
            "segments": result.segments
        }
```

### 2. Импортируйте в `src/models/__init__.py`

```python
from src.models.my_model_wrapper import MyModelWrapper
```

### 3. Используйте в TranscriptionStage через конфиг

```yaml
- name: transcription
  class: TranscriptionStage
  params:
    model: my-model-v1
```

---

## Советы

- Всегда документируйте, какие ключи ваш этап **читает** из `data` и **записывает** в `data`.
- Используйте `context["logger"]` для логирования вместо `print`.
- Для тяжёлых моделей реализуйте ленивую загрузку — загружайте в `run()`, а не в `__init__()`.
- Добавьте тест в `tests/` — шаблон смотрите в `tests/test_stage_template.py`.
