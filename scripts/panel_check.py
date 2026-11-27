"""母带面板“检测频段 A/B 对照”的真实执行证据与无浏览器自检。

它不使用任何固定/模拟数据：直接调用 dyneq.app._ab_compare，后者两次真实运行
dyneq.kernel.DynamicEQ。产出：
  out/panel_ab/panel_ab.json
  out/panel_ab/ab_<before>_to_<after>_{a,b,diff}.wav
  out/panel_ab/ab_compare.png

另会 build_app() 校验 Gradio 组件/事件绑定与输出数量可构建。
图形服务本身的 HTTP 检查见 README（启动后 curl 127.0.0.1:7860）。

用法（仓库根目录）:
    /tmp/venv/bin/python -I scripts/panel_check.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.io import wavfile

import matplotlib
matplotlib.use("Agg")

from dyneq import app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "panel_ab")
FS = app.FS

# 与面板默认值一致的一次主对照 + 四个快捷频带
CASES = [
    (2400.0, 3600.0),
    (2400.0, 800.0),
    (2400.0, 2000.0),
    (2400.0, 3000.0),
    (2400.0, 6000.0),
]


def _write_wav(path, audio_tuple):
    sr, sig = audio_tuple
    wavfile.write(path, sr, np.asarray(sig, dtype=np.float32))


def main():
    os.makedirs(OUT, exist_ok=True)

    built = app.build_app()
    build_ok = type(built).__name__ == "Blocks"
    print(f"build_app -> {type(built).__name__} (ok={build_ok})")

    evidence = {"build_app_ok": build_ok, "fs": FS, "cases": {}}
    main_fig = None
    for f0, f1 in CASES:
        fig, dry, a, b, diff, rep = app._ab_compare(
            "pilot_grid", None, f0, f1, 8.0, -32.0, True, 3000.0)
        r = json.loads(rep)
        tag = f"{int(f0)}_to_{int(f1)}"
        _write_wav(os.path.join(OUT, f"ab_{tag}_A_before.wav"), a)
        _write_wav(os.path.join(OUT, f"ab_{tag}_B_after.wav"), b)
        _write_wav(os.path.join(OUT, f"ab_{tag}_diff.wav"), diff)
        if main_fig is None:
            main_fig = fig
            fig.savefig(os.path.join(OUT, "ab_compare.png"), dpi=130)
        evidence["cases"][tag] = r
        d = r["output_difference_after_minus_before"]
        print(f"{tag}: before min {r['before']['gain_min_db']:6.2f} dB / "
              f"after min {r['after']['gain_min_db']:6.2f} dB | "
              f"max wet diff {d['max_abs']:.3e} audible={d['audible_change']}")

    # 固定处理频段（不跟随）路径也必须真实可跑
    fig2, *_rest, rep2 = app._ab_compare(
        "fixed_resonance", None, 1800.0, 3200.0, 10.0, -30.0, False, 2600.0)
    evidence["fixed_process_case"] = json.loads(rep2)
    fig2.savefig(os.path.join(OUT, "ab_fixed_process.png"), dpi=130)

    with open(os.path.join(OUT, "panel_ab.json"), "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"完成 -> {OUT}/panel_ab.json 与 WAV/PNG 证据")


if __name__ == "__main__":
    main()
