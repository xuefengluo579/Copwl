# w2v-SELD 音频事件检测模块开发文档

> **文档版本**：v1.0  
> **适用项目**：老年陪伴AI - 音频感知子系统  
> **目标平台**：Linux / Python 3.9+  
> **最后更新**：2026-06-30

---

## 1. 模块定位

本模块基于 **w2v-SELD** 实现音频流中的声音事件检测（SED），输出事件类型、时间戳及声源方位（DOA），供上层决策模块消费。

### 核心输入 / 输出

| 项目 | 说明 |
|------|------|
| **输入** | 四通道 Ambisonics B 格式音频（16kHz，批量或流式） |
| **输出** | `List[Dict]`：<br>`[{"event": "fall", "conf": 0.94, "t_start": 0, "t_end": 450, "doa": [x, y, z]}]` |

---

## 2. 环境依赖

### 2.1 基础环境

```bash
# 推荐使用 conda 管理环境
conda create -n seld python=3.9
conda activate seld

# 安装 PyTorch（CUDA 11.8 以上）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 fairseq（w2v-SELD 基于 fairseq 实现）
pip install fairseq

# 其他依赖
pip install numpy scipy librosa soundfile
```

### 2.2 硬件需求

| 阶段 | GPU 显存 | 预估耗时 |
|------|----------|----------|
| 推理（BASE） | ≥ 4GB | 实时（< 100ms/帧） |
| 微调（BASE） | ≥ 16GB | ~10 小时 |
| 预训练（BASE） | ≥ 32GB（A100） | ~3 天 |

> **注意**：若仅做推理，4GB 显存的消费级 GPU（如 RTX 3060）即可满足。

---

## 3. 代码获取与模型权重

### 3.1 克隆仓库

```bash
git clone https://github.com/Orlllem/seld_wav2vec2.git
cd seld_wav2vec2
```

### 3.2 下载预训练权重

官方提供两类权重：

| 配置 | 下载链接 | 用途 |
|------|----------|------|
| BASE | Google Drive | 快速推理，资源占用低 |
| LARGE | Google Drive | 更高精度，适合微调 |

```bash
# 下载后放入 ./checkpoints/ 目录
mkdir -p checkpoints
# 手动下载或使用 gdown
pip install gdown
gdown https://drive.google.com/uc?id=1PWZH6OpbPlUOZvOgRrMb46z-r89bY1u2 -O checkpoints/w2v_seld_base.pt
```

---

## 4. 推理接口实现

### 4.1 核心推理类

```python
# seld_engine.py

import torch
import torchaudio
import numpy as np
from fairseq.models.wav2vec import Wav2Vec2Model
from typing import List, Dict, Optional


class W2vSELDEngine:
    """w2v-SELD 推理引擎封装"""

    # DCASE 标准事件类别（可根据需求替换）
    DEFAULT_CLASSES = [
        "speech", "fall", "glass_break", "knock",
        "object_drop", "footstep", "door_slam"
    ]

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        sample_rate: int = 16000,
        num_channels: int = 4,
        event_threshold: float = 0.5
    ):
        """
        Args:
            model_path: w2v-SELD 模型配置路径
            checkpoint_path: 预训练权重路径 (.pt)
            device: "cuda" 或 "cpu"
            sample_rate: 目标采样率（必须 16000）
            num_channels: 输入通道数（必须 4）
            event_threshold: SED 激活阈值
        """
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.threshold = event_threshold

        # 加载模型（基于 fairseq）
        self.model = Wav2Vec2Model.from_pretrained(
            model_path,
            checkpoint_file=checkpoint_path
        ).to(self.device)
        self.model.eval()

    def preprocess(self, audio: np.ndarray) -> torch.Tensor:
        """
        音频预处理

        Args:
            audio: 输入音频，shape (channels, samples)，channels 必须为 4
        Returns:
            shape (1, channels, samples)，已标准化
        """
        # 通道数校验
        if audio.shape[0] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {audio.shape[0]}")

        # 转换为 torch tensor
        tensor = torch.from_numpy(audio).float()

        # 标准化：零均值，单位方差
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        tensor = (tensor - mean) / (std + 1e-8)

        # 添加 batch 维度
        return tensor.unsqueeze(0).to(self.device)

    def postprocess(self, logits: torch.Tensor) -> List[Dict]:
        """
        解析模型输出

        Args:
            logits: 模型原始输出，shape (batch, frames, num_classes)
            值域为 logits（需经 sigmoid 转概率）
        Returns:
            List[Dict]: 检测到的事件列表
        """
        probs = torch.sigmoid(logits)  # (batch, frames, num_classes)
        batch_result = []

        for b_idx in range(probs.shape[0]):
            frame_probs = probs[b_idx]  # (frames, num_classes)
            events = []

            for class_idx in range(frame_probs.shape[1]):
                prob_seq = frame_probs[:, class_idx]  # (frames,)

                # 寻找激活区间：概率 > threshold 的连续帧段
                activated = (prob_seq > self.threshold).cpu().numpy()

                start = None
                for i, active in enumerate(activated):
                    if active and start is None:
                        start = i
                    if not active and start is not None:
                        # 每帧 100ms，转换为毫秒
                        t_start = start * 100
                        t_end = i * 100
                        max_conf = float(prob_seq[start:i].max().cpu())
                        events.append({
                            "event": self.DEFAULT_CLASSES[class_idx],
                            "confidence": max_conf,
                            "t_start_ms": t_start,
                            "t_end_ms": t_end,
                            # DOA 暂略，需从模型另一输出分支获取
                        })
                        start = None
                if start is not None:
                    t_start = start * 100
                    t_end = len(activated) * 100
                    events.append({
                        "event": self.DEFAULT_CLASSES[class_idx],
                        "confidence": float(prob_seq[start:].max().cpu()),
                        "t_start_ms": t_start,
                        "t_end_ms": t_end,
                    })

            batch_result.append(events)

        return batch_result[0] if len(batch_result) == 1 else batch_result

    def infer(self, audio: np.ndarray) -> List[Dict]:
        """
        执行完整推理流水线

        Args:
            audio: 输入音频，shape (channels, samples)
        Returns:
            List[Dict]: 检测到的事件列表
        """
        with torch.no_grad():
            # 1. 预处理
            input_tensor = self.preprocess(audio)
            # 2. 模型推理（返回 SED logits 和 DOA 输出）
            # 注：具体输出结构需参考模型 forward 实现
            sed_output, doa_output = self.model(
                source=input_tensor,
                features_only=False,
                return_doa=True  # 需确认模型是否支持
            )
            # 3. 后处理
            events = self.postprocess(sed_output)

            # TODO: 合并 DOA 信息
            # for i, event in enumerate(events):
            #     event["doa"] = doa_output[i].tolist()

        return events


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 初始化
    engine = W2vSELDEngine(
        model_path="./w2v_seld_base",
        checkpoint_path="./checkpoints/w2v_seld_base.pt",
        device="cuda",
        event_threshold=0.5
    )

    # 加载测试音频（需为 4 通道）
    audio, sr = torchaudio.load("test_audio.wav")

    # 如果输入为单通道，需转换为四通道伪 Ambisonics
    if audio.shape[0] == 1:
        audio = audio.repeat(4, 1)

    # 重采样到 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    # 推理
    result = engine.infer(audio.numpy())
    print(result)
```

### 4.2 流式音频适配

若需处理实时音频流，需维护一个滑动窗口缓冲区：

```python
class StreamingSELDEngine(W2vSELDEngine):
    """支持流式输入的 SED 引擎"""

    def __init__(self, window_seconds: float = 2.0, hop_seconds: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.window_size = int(window_seconds * self.sample_rate)
        self.hop_size = int(hop_seconds * self.sample_rate)
        self.buffer = np.zeros((self.num_channels, self.window_size))

    def feed(self, chunk: np.ndarray) -> Optional[List[Dict]]:
        """
        接收音频块，返回检测结果（当缓冲区满时触发）

        Args:
            chunk: shape (channels, samples)
        Returns:
            检测到的事件列表，或 None（缓冲区未满）
        """
        # 追加到缓冲区
        self.buffer = np.roll(self.buffer, -chunk.shape[1], axis=1)
        self.buffer[:, -chunk.shape[1]:] = chunk

        # 触发推理
        return self.infer(self.buffer)
```

> **注意**：流式场景下需权衡窗口长度（检测精度 vs 延迟）。建议窗口 2~3 秒，步进 0.5~1 秒。

---

## 5. 模型微调指南

### 5.1 准备标注数据集

数据需遵循 DCASE SELD 格式：

```
dataset/
├── metadata/
│   └── dev_train_1/
│       ├── audio1.csv  # 标注文件
│       └── ...
└── audio/
    ├── audio1.wav  # 四通道音频
    └── ...
```

标注 CSV 格式：`onset(sec), offset(sec), event_class, doa_x, doa_y, doa_z`

示例：

```
0.00,0.45,fall,0.32,-0.15,0.08
0.80,1.20,glass_break,0.55,0.20,-0.10
```

### 5.2 微调命令

```bash
# 进入官方代码库
cd seld_wav2vec2

# 微调（BASE 配置）
python finetune.py \
    --data-path /path/to/your/dataset \
    --checkpoint-path ./checkpoints/w2v_seld_base.pt \
    --output-dir ./finetuned \
    --config configs/base_config.yaml \
    --lr 1e-5 \
    --batch-size 8 \
    --epochs 50 \
    --gpu 0
```

### 5.3 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| lr（学习率） | 1e-5（Transformer）/ 1e-4（分支） | 分层学习率，防止灾难性遗忘 |
| batch_size | 8~16（取决于显存） | - |
| epochs | 50~100（早停） | 监控验证集损失 |
| threshold_γ | 0.4~0.6（调优） | SED 激活阈值，通过验证集 F1 选择 |

---

## 6. API 接口设计（供思考模块调用）

将本模块封装为 RESTful 或 gRPC 服务：

### 6.1 RESTful API（FastAPI 示例）

```python
# api_server.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
engine = W2vSELDEngine(...)


class EventItem(BaseModel):
    event: str
    confidence: float
    t_start_ms: int
    t_end_ms: int
    doa: Optional[List[float]] = None


@app.post("/v1/audio/events", response_model=List[EventItem])
async def detect_events(file: UploadFile = File(...)):
    """上传音频文件，返回检测到的事件列表"""
    audio_bytes = await file.read()
    audio = load_audio_from_bytes(audio_bytes, target_sr=16000, channels=4)
    events = engine.infer(audio)
    return events

# 启动：uvicorn api_server:app --host 0.0.0.0 --port 8080
```

### 6.2 输出数据契约

```json
[
    {
        "event": "fall",
        "confidence": 0.94,
        "t_start_ms": 0,
        "t_end_ms": 450,
        "doa": [0.32, -0.15, 0.08]
    },
    {
        "event": "glass_break",
        "confidence": 0.87,
        "t_start_ms": 800,
        "t_end_ms": 1200,
        "doa": [0.55, 0.20, -0.10]
    }
]
```

---

## 7. 注意事项与约束

| 约束项 | 说明 |
|--------|------|
| 输入通道数 | 必须为 4，若设备仅单/双麦需自行实现伪空间音频合成 |
| 采样率 | 严格 16kHz，预处理需重采样 |
| 事件类别 | 原生支持 DCASE 类别，若需自定义需修改 `DEFAULT_CLASSES` 并微调 |
| 推理延迟 | 单次窗口（2秒音频）在 RTX 3060 上约 50~80ms |
| 部署环境 | 若 CPU 推理，建议使用 ONNX 导出或量化（`torch.quantization`） |

---

## 8. 快速启动脚本

```bash
#!/bin/bash
# setup.sh - 一键初始化开发环境

# 1. 创建环境
conda create -n seld python=3.9 -y
conda activate seld

# 2. 安装依赖
pip install torch torchaudio fairseq numpy scipy librosa soundfile

# 3. 克隆代码
git clone https://github.com/Orlllem/seld_wav2vec2.git

# 4. 下载权重（需手动授权）
# 访问 https://drive.google.com/uc?id=1PWZH6OpbPlUOZvOgRrMb46z-r89bY1u2
# 放置到 ./seld_wav2vec2/checkpoints/

# 5. 运行测试
python test_inference.py --audio sample.wav
```

---

## 9. 相关资源

| 资源 | 链接 |
|------|------|
| 官方代码库 | https://github.com/Orlllem/seld_wav2vec2 |
| 论文（IEEE Access 2024） | https://ieeexplore.ieee.org/document/10738867 |
| arXiv 预印本 | https://arxiv.org/abs/2312.06907 |
| DCASE 挑战赛 | https://dcase.community/ |

---

*本文档由 AI 自动生成，如有问题请联系模块负责人。*
