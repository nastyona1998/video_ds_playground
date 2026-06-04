# Архитектура Video DS Playground

## Принципы

1. **Каждый этап — класс**, наследующий абстрактный `Stage`.
2. **Данные передаются словарём** `data: dict` между этапами. Каждый этап дополняет его новыми ключами.
3. **Конфигурация через YAML** — этапы, модели и параметры задаются в файлах конфигурации.
4. **Реестр** позволяет добавлять новые модели и этапы без изменения ядра.
5. **Кеширование промежуточных результатов** — при повторном запуске пайплайн пропускает уже выполненные этапы.

---

## Поток данных

```
video_path (str)
    │
    ▼
[FetchStage]          → data["video_path"] (локальный путь после скачивания)
    │
    ▼
[AudioExtractStage]   → data["audio_path"], data["audio_meta"]
    │
    ▼
[VADStage]            → data["speech_intervals"], data["silence_intervals"]
    │
    ▼
[DenoiseStage]        → data["denoised_audio_path"]   (опционально)
    │
    ▼
[TranscriptionStage]  → data["transcription"], data["word_timestamps"], data["language"]
    │
    ▼
[DiarizationStage]    → data["speaker_turns"], data["speaker_transcription"]
    │
    ▼
[QualityMetricsStage] → data["metrics"] {pesq, stoi, snr, accent_score}
    │
    ▼
[PostprocessStage]    → data["clean_transcription"], data["removed_fragments"]
    │
    ▼
[ReportStage]         → data["report_path"]
```

---

## Классы ядра

### Stage (core/base.py)

```python
class Stage(ABC):
    def run(self, data: dict, context: dict) -> dict: ...
    def get_config_schema(self) -> dict: ...
```

- `data` — словарь, накапливающий результаты этапов.
- `context` — общие ресурсы (логгер, кеш, конфиг пайплайна).

### ModelWrapper (core/base.py)

```python
class ModelWrapper(ABC):
    def load(self, **kwargs): ...
    def predict(self, audio_path: str, **params) -> Any: ...
```

### Pipeline (core/pipeline.py)

```python
class Pipeline:
    def __init__(self, config_path: str): ...
    def run(self, video_path: str) -> dict: ...
    def add_stage(self, stage: Stage, position: int = None): ...
    def skip_stage(self, stage_name: str): ...
```

### Registry (core/registry.py)

```python
STAGE_REGISTRY: dict[str, Type[Stage]] = {}
MODEL_REGISTRY: dict[str, Type[ModelWrapper]] = {}

def register_stage(name: str): ...   # декоратор
def register_model(name: str): ...   # декоратор
```

---

## Конфигурация (YAML)

```yaml
pipeline:
  name: "whisper_experiment"
  cache_dir: "data/.cache"
  report_output: "data/reports/"

  stages:
    - name: fetch
      class: FetchStage
      params:
        cloud: nextcloud
        remote_path: "/videos/lecture.mp4"

    - name: audio_extract
      class: AudioExtractStage
      params:
        sample_rate: 16000
        channels: 1
        format: wav

    - name: vad
      class: VADStage
      params:
        model: silero
        threshold: 0.5

    - name: transcription
      class: TranscriptionStage
      params:
        model: whisper-large-v3
        language: ru

    - name: diarization
      class: DiarizationStage
      params:
        model: pyannote
        num_speakers: null   # auto-detect

    - name: quality_metrics
      class: QualityMetricsStage
      params:
        metrics: [pesq, stoi, snr]

    - name: postprocess
      class: PostprocessStage
      params:
        remove_hallucinations: true
        remove_repetitions: true

    - name: report
      class: ReportStage
      params:
        format: html
```
