"""w2v-SELD 音频事件检测模块 — setuptools 安装配置."""

from setuptools import setup, find_packages

setup(
    name="w2v-seld",
    version="0.1.0",
    description="基于 wav2vec 2.0 的声音事件检测与定位 (SELD) 推理引擎",
    author="老年陪伴AI 团队",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.6.0",
        "PyYAML>=6.0",
        "python-multipart>=0.0.9",
        "gdown>=5.2.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-asyncio>=0.23.0",
            "pytest-benchmark>=4.0.0",
            "httpx>=0.27.0",
            "black",
            "ruff",
        ],
    },
    entry_points={
        "console_scripts": [
            "seld-api=seld.api:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
