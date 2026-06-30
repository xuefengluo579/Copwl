# ============================================================
# w2v-SELD 一键环境初始化脚本 (Windows PowerShell)
# ============================================================
param(
    [string]$CondaEnv = "seld",
    [string]$PythonVersion = "3.9"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " w2v-SELD 环境初始化 (Windows)" -ForegroundColor Cyan
Write-Host " 项目目录: $ProjectDir" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# ── 1. Conda 环境 ──────────────────────────────────────────
$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($conda) {
    $envList = conda env list 2>$null
    if ($envList -match $CondaEnv) {
        Write-Host "[✓] Conda 环境 '$CondaEnv' 已存在" -ForegroundColor Green
    } else {
        Write-Host "[→] 创建 conda 环境: $CondaEnv (Python $PythonVersion)" -ForegroundColor Yellow
        conda create -n $CondaEnv python=$PythonVersion -y
    }
    Write-Host "[→] 激活 conda 环境..." -ForegroundColor Yellow
    conda activate $CondaEnv
} else {
    Write-Host "[!] 未检测到 conda，使用系统 Python" -ForegroundColor Red
    Write-Host "    推荐安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
}

# ── 2. CUDA 检测 ──────────────────────────────────────────
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "[✓] CUDA GPU 检测到" -ForegroundColor Green
    $TorchIndex = "https://download.pytorch.org/whl/cu118"
} else {
    Write-Host "[!] 未检测到 CUDA GPU，安装 CPU 版 PyTorch" -ForegroundColor Yellow
    $TorchIndex = "https://download.pytorch.org/whl/cpu"
}

# ── 3. PyTorch ────────────────────────────────────────────
Write-Host "[→] 安装 PyTorch + torchaudio..." -ForegroundColor Yellow
pip install torch==2.1.0 torchaudio==2.1.0 --index-url $TorchIndex

# ── 4. 其他依赖 ──────────────────────────────────────────
Write-Host "[→] 安装项目依赖..." -ForegroundColor Yellow
Set-Location $ProjectDir
pip install -e ".[dev]"

# ── 5. fairseq ────────────────────────────────────────────
Write-Host "[→] 安装 fairseq..." -ForegroundColor Yellow
try {
    pip install fairseq
} catch {
    Write-Host "[!] PyPI fairseq 不可用，从 GitHub 安装..." -ForegroundColor Yellow
    pip install git+https://github.com/pytorch/fairseq.git@v0.12.2
}

# ── 6. 下载模型权重 ───────────────────────────────────────
Write-Host ""
Write-Host "[→] 下载预训练权重..." -ForegroundColor Yellow
try {
    python scripts/download_weights.py
} catch {
    Write-Host "[!] 权重下载失败，请稍后手动运行: python scripts/download_weights.py" -ForegroundColor Red
}

# ── 7. 验证 ────────────────────────────────────────────────
Write-Host ""
Write-Host "[→] 验证安装..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
try {
    python -c "import seld; print(f'seld {seld.__version__}')"
    Write-Host "[✓] seld 包导入成功" -ForegroundColor Green
} catch {
    Write-Host "[!] seld 包导入失败: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " 安装完成!" -ForegroundColor Green
Write-Host " - 激活环境: conda activate $CondaEnv" -ForegroundColor White
Write-Host " - 启动 API: seld-api" -ForegroundColor White
Write-Host " - 运行测试: pytest -m fast" -ForegroundColor White
Write-Host "==============================================" -ForegroundColor Cyan
