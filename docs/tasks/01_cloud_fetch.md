# Задание 01: Загрузка видео из облачного хранилища

## Цель

Научиться подключаться к облачному хранилищу (Nextcloud / S3) и скачивать видеофайлы в локальную директорию через абстрактный адаптер.

---

## Теоретическая справка

### WebDAV и Nextcloud

Nextcloud использует протокол **WebDAV** для доступа к файлам. Это расширение HTTP, которое добавляет методы `PROPFIND`, `MKCOL`, `COPY`, `MOVE` к стандартным `GET`/`PUT`.

- [RFC 4918 — WebDAV](https://www.rfc-editor.org/rfc/rfc4918)
- [Nextcloud WebDAV docs](https://docs.nextcloud.com/server/latest/user_manual/en/files/access_webdav.html)
- [webdavclient3 на PyPI](https://pypi.org/project/webdavclient3/)

### Amazon S3

Amazon S3 (и совместимые хранилища: MinIO, Yandex Object Storage) используют REST API. Библиотека `boto3` предоставляет Python-интерфейс.

- [boto3 S3 quickstart](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-examples.html)

### Паттерн Адаптер

Для того чтобы пайплайн не зависел от конкретного хранилища, используется паттерн **Adapter**: `CloudClient` — абстрактный класс, `NextcloudClient` и `S3Client` — конкретные реализации.

---

## Структура кода

Откройте файлы:
- `src/core/cloud_client.py` — абстрактный класс и реализации
- `src/stages/fetch.py` — этап пайплайна
- `config/default_pipeline.yaml` — конфигурация

---

## Задание

### Шаг 1. Изучите абстрактный класс

```python
# src/core/cloud_client.py
class CloudClient(ABC):
    @abstractmethod
    def download(self, remote_path: str, local_path: str) -> str: ...
    
    @abstractmethod
    def list_files(self, remote_dir: str) -> list[str]: ...
    
    @abstractmethod
    def upload(self, local_path: str, remote_path: str) -> bool: ...
```

### Шаг 2. Реализуйте `NextcloudClient`

Заполните метод `download()` в `src/core/cloud_client.py`. Используйте `webdavclient3`.

```python
# Подсказка: инициализация клиента
options = {
    "webdav_hostname": self.host,
    "webdav_login": self.username,
    "webdav_password": self.password,
}
client = webdav3.client.Client(options)
client.download_sync(remote_path=remote_path, local_path=local_path)
```

### Шаг 3. Запустите этап

Настройте `.env` (см. `.env.example`) и запустите:

```bash
python run.py --config config/default_pipeline.yaml --video nextcloud://videos/sample.mp4 --stages fetch
```

### Шаг 4 (опционально). Реализуйте `S3Client`

Повторите для Amazon S3 / MinIO, используя `boto3`.

---

## Ожидаемый результат

- Видеофайл скачан в `data/raw_videos/`
- В логе: имя файла, размер, время скачивания

---

## Дополнительные вызовы

- Добавьте проверку: если файл уже скачан, пропустить загрузку (кеширование).
- Реализуйте `list_files()` и напишите скрипт, который скачивает все `.mp4` из указанной директории.
- Добавьте прогресс-бар с `tqdm`.
