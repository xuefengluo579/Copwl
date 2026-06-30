#!/bin/bash
# ============================================================
# w2v-SELD 微调 — 魔搭 GPU 一键执行脚本
#
# 在魔搭终端里直接运行:
#   bash finetune_on_modelscope.sh
# ============================================================
set -e

echo "=============================================="
echo " w2v-SELD 微调 — 魔搭 GPU 环境"
echo "=============================================="

# ── 1. 环境检查 ──────────────────────────────────────────
nvidia-smi | head -3
echo ""

# ── 2. 克隆仓库 ──────────────────────────────────────────
cd /mnt/workspace

echo "[1/6] 克隆项目代码..."
if [ ! -d "Copwl" ]; then
    git clone --depth 1 https://github.com/xuefengluo579/Copwl.git
fi

echo "[2/6] 克隆 seld_wav2vec2 (训练框架)..."
if [ ! -d "seld_wav2vec2" ]; then
    git clone --depth 1 https://github.com/Orlllem/seld_wav2vec2.git
fi

# ── 3. 安装依赖 ──────────────────────────────────────────
echo "[3/6] 安装 PyTorch (CUDA)..."
pip install torch==2.1.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121 -q

echo "[4/6] 安装 fairseq + 其他依赖..."
pip install git+https://github.com/pytorch/fairseq.git@v0.12.2 -q
pip install numpy scipy librosa soundfile pydantic PyYAML gdown tqdm -q

cd /mnt/workspace/Copwl/音频识别
pip install -e . -q 2>/dev/null || true

# ── 4. 下载预训练权重 ─────────────────────────────────────
echo "[5/6] 下载 w2v-SELD BASE 预训练权重 (约 400MB)..."
mkdir -p checkpoints
if [ ! -f "checkpoints/w2v_seld_base.pt" ]; then
    gdown "https://drive.google.com/uc?id=1PWZH6OpbPlUOZvOgRrMb46z-r89bY1u2" \
        -O checkpoints/w2v_seld_base.pt || {
        echo "⚠️ gdown 失败。请在本地下载权重后通过魔搭数据集上传。"
        echo "   上传后放到 /mnt/workspace/Copwl/音频识别/checkpoints/"
    }
else
    echo "  ✓ 权重已存在"
fi

# ── 5. 准备数据 ──────────────────────────────────────────
echo "[6/6] 准备训练数据..."

# ============================================================
# 在这里指定你的训练数据路径
# 预期结构: /mnt/workspace/dataset/
#   ├── annotations.csv
#   ├── scene_001.wav
#   ├── scene_002.wav
#   └── ...
#
# 标注格式 (每行):
#   filename,onset,offset,event_class
#   例如: scene_001.wav,0.5,1.2,fall
# ============================================================

DATA_SOURCE="/mnt/workspace/my_audio_data"      # ← 改成你的数据路径
OUTPUT_DATASET="/mnt/workspace/dataset_seld"

if [ -d "$DATA_SOURCE" ]; then
    echo "  数据源: $DATA_SOURCE"
    python scripts/prepare_dataset.py "$DATA_SOURCE" \
        -o "$OUTPUT_DATASET" \
        --sr 16000 --channels 4
else
    echo "  ⚠️ 数据目录不存在: $DATA_SOURCE"
    echo "  跳过数据准备。上传数据后重新运行:"
    echo "    python scripts/prepare_dataset.py <your_data_dir> -o $OUTPUT_DATASET"
fi

# ── 训练 ──────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " 环境准备完成。数据就绪后，运行以下命令开始训练:"
echo "=============================================="
echo ""
echo "cd /mnt/workspace/seld_wav2vec2"
echo ""
echo "python finetune.py \\"
echo "    --data-path $OUTPUT_DATASET \\"
echo "    --checkpoint-path /mnt/workspace/Copwl/音频识别/checkpoints/w2v_seld_base.pt \\"
echo "    --output-dir /mnt/workspace/finetuned \\"
echo "    --config configs/base_config.yaml \\"
echo "    --lr 1e-5 \\"
echo "    --batch-size 8 \\"
echo "    --epochs 50 \\"
echo "    --gpu 0"
echo ""
echo "=============================================="
echo " 训练完后，把权重放到推理目录:"
echo "   cp /mnt/workspace/finetuned/best.pt \\"
echo "      /mnt/workspace/Copwl/音频识别/checkpoints/finetuned.pt"
echo ""
echo " 然后在推理时使用:"
echo "   export SELD_CHECKPOINT_PATH=checkpoints/finetuned.pt"
echo "   python -c 'from seld import W2vSELDEngine; ...'"
echo "=============================================="
