"""
Обёртки для методов шумоподавления.

Установка:
    pip install noisereduce soundfile
    pip install speechbrain   # для SpeechBrain
"""
from __future__ import annotations

from pathlib import Path

from src.core.base import ModelWrapper
from src.core.registry import register_model


@register_model("noisereduce")
class NoiseReduceWrapper(ModelWrapper):
    """
    Шумоподавление методом спектрального вычитания.
    Быстро, не требует GPU, но агрессивный режим может искажать речь.

    Задание: docs/tasks/07_denoising.md
    """

    def load(self, **kwargs) -> None:
        try:
            import noisereduce  # noqa
            import soundfile  # noqa
        except ImportError:
            raise ImportError("Установите: pip install noisereduce soundfile")
        # Нет весов для загрузки
        self.model = True

    def predict(
        self,
        audio_path: str,
        output_path: str | None = None,
        prop_decrease: float = 1.0,
        stationary: bool = False,
        n_std_thresh_stationary: float = 1.5,
        **kwargs,
    ) -> dict:
        """
        TODO: Реализуйте шумоподавление.

        Подсказка:
            import noisereduce as nr
            import soundfile as sf
            import numpy as np

            data, sr = sf.read(audio_path)
            reduced = nr.reduce_noise(
                y=data,
                sr=sr,
                prop_decrease=prop_decrease,
                stationary=stationary,
                n_std_thresh_stationary=n_std_thresh_stationary,
            )
            out = output_path or audio_path.replace(".wav", "_nr.wav")
            sf.write(out, reduced, sr)
            return {"output_path": out}
        """
        import noisereduce as nr
        import soundfile as sf

        data, sr = sf.read(audio_path)
        reduced = nr.reduce_noise(
            y=data,
            sr=sr,
            prop_decrease=prop_decrease,
            stationary=stationary,
            n_std_thresh_stationary=n_std_thresh_stationary,
        )
        out = output_path or audio_path.replace(".wav", "_nr.wav")
        sf.write(out, reduced, sr)
        return {"output_path": out}


@register_model("speechbrain-sepformer")
class SpeechBrainDenoiser(ModelWrapper):
    """
    Нейросетевое шумоподавление через SpeechBrain SepFormer.
    Высокое качество, требует GPU для разумной скорости.

    Установка: pip install speechbrain
    """

    def load(self, savedir: str = "data/.cache/speechbrain", **kwargs) -> None:
        try:
            from speechbrain.pretrained import SepformerSeparation
        except ImportError:
            raise ImportError("Установите: pip install speechbrain")
        self.model = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wham16k-enhancement",
            savedir=savedir,
        )

    def predict(self, audio_path: str, output_path: str | None = None, **kwargs) -> dict:
        import torchaudio
        out = output_path or str(Path(audio_path).with_suffix("")) + "_sb.wav"

        est_sources = self.model.separate_file(path=audio_path)
        # est_sources: [batch, time, sources] — берём первый источник (речь)
        torchaudio.save(out, est_sources[:, :, 0].detach().cpu(), 8000)
        return {"output_path": out}


@register_model("demucs")
class DemucsWrapper(ModelWrapper):
    """
    Шумоподавление с помощью Facebook Demucs (разделение источников).

    Установка: pip install demucs
    """

    def load(self, model_name: str = "htdemucs", **kwargs) -> None:
        try:
            import demucs.api  # noqa
        except ImportError:
            raise ImportError("Установите: pip install demucs")
        self._model_name = model_name

    def predict(self, audio_path: str, output_path: str | None = None, **kwargs) -> dict:
        import subprocess
        out_dir = str(Path(audio_path).parent / "demucs_out")
        subprocess.run(
            ["python", "-m", "demucs", "--two-stems=vocals",
             "-n", self._model_name, audio_path, "-o", out_dir],
            check=True,
        )
        vocals_path = str(
            Path(out_dir) / self._model_name / Path(audio_path).stem / "vocals.wav"
        )
        if output_path:
            import shutil
            shutil.move(vocals_path, output_path)
            vocals_path = output_path
        return {"output_path": vocals_path}
