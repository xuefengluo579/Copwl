"""
w2v-SELD 推理演示 — 用合成音频跑一遍完整流水线。

运行:
    python run_inference_demo.py

注意: 此脚本需要 fairseq + 模型权重。
      如果本地没有 GPU/fairseq，在魔搭 GPU 上运行。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main():
    # 检查 fairseq 是否可用
    try:
        import fairseq  # noqa: F401
    except ImportError:
        print("=" * 55)
        print(" ⚠️ fairseq 未安装，无法在本地运行。")
        print("   请在魔搭 GPU 终端运行此脚本。")
        print("   魔搭上的命令:")
        print("     cd /mnt/workspace/Copwl/音频识别")
        print("     python run_inference_demo.py")
        print("=" * 55)
        return

    print("=" * 55)
    print(" w2v-SELD 推理演示")
    print("=" * 55)

    import torch
    from seld import SELDConfig, W2vSELDEngine

    # 1. 设备检测
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥 设备: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB")

    # 2. 加载配置
    config = SELDConfig.from_yaml("config/default.yaml")
    config.model.device = device

    # 3. 加载引擎
    print("\n📦 加载模型...")
    import time
    t0 = time.time()

    try:
        engine = W2vSELDEngine(config)
        print(f"   ✅ 加载完成 ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"   ❌ 加载失败: {e}")
        print(f"\n   请先下载权重:")
        print(f"     python scripts/download_weights.py --variant base")
        return

    # 4. 生成测试音频 (4通道, 3秒, 包含模拟事件)
    print("\n🎵 生成测试音频 (3秒, 4通道, 包含模拟事件)...")
    from tests.fixtures.synthetic import generate_event_audio
    audio = generate_event_audio(
        sample_rate=16000,
        duration_sec=3.0,
        events=[
            (0.5, 1.0, 0),  # speech @ 0.5s
            (1.5, 1.8, 3),  # knock @ 1.5s
            (2.2, 2.6, 1),  # fall @ 2.2s
        ],
        seed=42,
    )
    print(f"   形状: {audio.shape}, RMS: {audio.std():.3f}")

    # 5. 推理
    print("\n🔍 推理中...")
    events, elapsed = engine.infer(audio)
    print(f"   耗时: {elapsed:.1f}ms")

    # 6. 结果
    print(f"\n📊 检测到 {len(events)} 个事件:")
    for e in events:
        emoji = {"fall": "🚨", "glass_break": "💥", "knock": "👊"}.get(
            e.event.value, "📢"
        )
        print(
            f"   {emoji} {e.event.value:15s} "
            f"conf={e.confidence:.3f}  "
            f"t=[{e.t_start_ms}..{e.t_end_ms}]ms  "
            f"dur={e.duration_ms}ms"
        )

    # 7. 基准测试
    print("\n⏱ 性能基准 (GPU 预热后):")
    for dur in [1.0, 2.0, 5.0]:
        test_audio = np.random.randn(4, int(dur * 16000)).astype(np.float32)
        _, e = engine.infer(test_audio)
        rtf = e / (dur * 1000)  # 实时率: <1 = 比实时快
        print(f"   {dur:.0f}s 音频: {e:.1f}ms (RTF={rtf:.3f}, {'✅ 实时' if rtf<1 else '⚠️ 慢于实时'})")

    print("\n" + "=" * 55)
    print(" 推理演示完成")
    print("=" * 55)


if __name__ == "__main__":
    main()
