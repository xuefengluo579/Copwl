#!/bin/bash
# ============================================================
# w2v-SELD 一键环境初始化脚本 (Linux / macOS)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo " w2v-SELD 环境初始化"
echo " 项目目录: $PROJECT_DIR"
echo "=============================================="

# ── 1. Conda 环境 ──────────────────────────────────────────
ENV_NAME="${SELD_CONDA_ENV:-seld}"
PYTHON_VERSION="${SELD_PYTHON_VERSION:-3.9}"

if command -v conda &> /dev/null; then
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "[✓] Conda 环境 '${ENV_NAME}' 已存在"
    else
        echo "[→] 创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})"
        conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    fi
    echo "[→] 激活 conda 环境..."
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "[!] 未检测到 conda，使用系统 Python"
    echo "    推荐安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
fi

# ── 2. CUDA 检测 ──────────────────────────────────────────
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
    echo "[✓] CUDA 可用 (版本 ${CUDA_VERSION})"
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    echo "[!] 未检测到 CUDA GPU，安装 CPU 版 PyTorch"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# ── 3. PyTorch ────────────────────────────────────────────
echo "[→] 安装 PyTorch + torchaudio..."
pip install torch==2.1.0 torchaudio==2.1.0 --index-url "${TORCH_INDEX}"

# ── 4. 其他依赖 ──────────────────────────────────────────
echo "[→] 安装项目依赖..."
cd "$PROJECT_DIR"
pip install -e ".[dev]"

# ── 5. fairseq ────────────────────────────────────────────
echo "[→] 安装 fairseq..."
pip install fairseq 2>/dev/null || {
    echo "[!] PyPI fairseq 不可用，从 GitHub 安装..."
    pip install git+https://github.com/pytorch/fairseq.git@v0.12.2
}

# ── 6. 下载模型权重 ───────────────────────────────────────
echo ""
echo "[→] 下载预训练权重..."
python scripts/download_weights.py || {
    echo "[!] 权重下载失败，请稍后手动运行: python scripts/download_weights.py"
}

# ── 7. 验证 ────────────────────────────────────────────────
echo ""
echo "[→] 验证安装..."
python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import seld; print(f'seld {seld.__version__}')"

echo ""
echo "=============================================="
echo " 安装完成!"
echo " - 激活环境: conda activate ${ENV_NAME}"
echo " - 启动 API: seld-api"
echo " - 运行测试: pytest -m fast"
echo "=============================================="
