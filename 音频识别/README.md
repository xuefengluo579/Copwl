# w2v-SELD 音频事件检测模块

> **老年陪伴AI** — 音频感知子系统

基于 wav2vec 2.0 预训练模型的声音事件检测与定位 (Sound Event Detection and Localization, SELD) 推理引擎。

## 功能

- 从四通道 Ambisonics B 格式音频中检测 7 类声音事件：
  - 💬 speech（语音）
  - 🦯 fall（跌倒）
  - 💥 glass_break（玻璃破碎）
  - 👊 knock（敲门）
  - 📦 object_drop（物体掉落）
  - 👣 footstep（脚步声）
  - 🚪 door_slam（关门声）
- 输出：事件类型、置信度、时间戳、声源方位 (DOA)
- 支持批量推理和流式音频输入

## 快速开始

### 1. 环境初始化

```bash
# Linux / macOS
bash scripts/setup.sh

# Windows
powershell -File scripts/setup.ps1
```

### 2. 安装

```bash
pip install -e .
```

### 3. 下载预训练权重

```bash
python scripts/download_weights.py
```

### 4. 推理示例

```python
from seld import W2vSELDEngine, SELDConfig

config = SELDConfig.from_yaml("config/default.yaml")
engine = W2vSELDEngine(config)

import torchaudio
audio, sr = torchaudio.load("sample.wav")
events = engine.infer(audio.numpy())
print(events)
```

### 5. 启动 API 服务

```bash
seld-api
# 或: python -m seld.api
```

访问 http://localhost:8080/docs 查看交互式 API 文档。

## 项目结构

```
src/seld/           # Python 包
tests/              # 测试套件
config/             # 配置文件
scripts/            # 部署脚本
checkpoints/        # 模型权重（需手动下载）
```

## 依赖

- Python >= 3.9
- PyTorch >= 2.1 (CUDA 推荐)
- fairseq (wav2vec 2.0 实现)
- FastAPI + uvicorn (API 服务)

详见 [requirements.txt](./requirements.txt)

## 相关资源

- [官方代码库](https://github.com/Orlllem/seld_wav2vec2)
- [论文 (IEEE Access 2024)](https://ieeexplore.ieee.org/document/10738867)
- [DCASE 挑战赛](https://dcase.community/)
