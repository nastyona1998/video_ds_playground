from setuptools import setup, find_packages

setup(
    name="video_ds_playground",
    version="0.1.0",
    description="Учебный полигон для обработки видео: транскрибация, диаризация, VAD, шумоподавление",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pyyaml>=6.0",
        "python-dotenv>=1.0",
        "ffmpeg-python>=0.2.0",
        "soundfile>=0.12",
        "jinja2>=3.1",
        "jiwer>=3.0",
        "tqdm>=4.66",
    ],
    entry_points={
        "console_scripts": [
            "vdsp=run:main",
        ],
    },
)
