"""母带试听面板：由动态均衡内核驱动的 Gradio 界面。

启动（在仓库根目录）:
    /tmp/venv/bin/python -I -m dyneq.app
然后浏览器打开终端里打印的地址（默认 http://127.0.0.1:7860）。

控件由内核 schema 自动生成：数值参数用 Number 输入（不拦截越界提交，
越界/非法值由内核钳位或拒绝，物理原因反馈到"内核反馈"区），布尔用
Checkbox。结果区呈现动态增益曲线、检测/处理中心轨迹、响应-模型对照，
以及干声/输出/检测通路三段可试听音频。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FONT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "fonts", "NotoSansSC-Regular.otf")
if os.path.exists(_FONT):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

import gradio as gr
from scipy.signal import resample_poly

from dyneq import stimuli
from dyneq.kernel import DynamicEQ, model_response_db
from dyneq.render import pilot_gain_db

FS = 48000
MAX_DUR_S = 12.0
STIM_NAMES = {
    "glide": "滑移：原共振滑向邻峰",
    "reonset": "重新起振：原音继续+邻音起振",
    "fixed_resonance": "固定音+可变窄带共振",
    "pilot_grid": "导频栅格+触发共振",
}
MEAS_FREQS = (500.0, 997.0, 2400.0, 3400.0, 3600.0, 6000.0)


def _schema():
    return DynamicEQ(FS).schema()


def _load_stimulus(name, upload):
    if upload is not None:
        sr, data = upload
        x = np.asarray(data, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if np.issubdtype(np.asarray(data).dtype, np.integer):
            x = x / float(np.iinfo(np.asarray(data).dtype).max)
        if sr != FS:
            g = np.gcd(sr, FS)
            x = resample_poly(x, FS // g, sr // g)
        return x[: int(MAX_DUR_S * FS)], f"上传音频（{sr}Hz 重采样到 {FS}Hz）"
    x, meta = stimuli.STIMULI[name](FS)
    return x, STIM_NAMES[name]


def _gain_figure(taps, kernel):
    log = taps["log"]
    t = log["t"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax = axes[0]
    ax.plot(t, log["gain_db"], color="C3", lw=1.2, label="动态增益 dB")
    ax.plot(t, log["env_db"], color="C0", lw=0.8, alpha=0.7, label="检测包络 dB")
    ax.axhline(kernel.params["threshold_db"], ls=":", c="gray", lw=0.8, label="阈值")
    ax.set_ylabel("dB")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("动态增益与检测包络")
    ax = axes[1]
    ax.plot(t, log["f_detect"], color="C0", label="检测中心")
    ax.plot(t, log["f_process"], color="C3", ls="--", label="处理中心")
    ax.set_ylabel("Hz")
    ax.set_xlabel("时间 s")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("检测/处理中心轨迹（检测历史与动态滤波器同步）")
    fig.tight_layout()
    return fig


def _response_figure(x, y, taps, kernel):
    log = taps["log"]
    i_max = int(np.argmin(log["gain_db"]))
    g = float(log["gain_db"][i_max])
    fp = float(log["f_process"][i_max])
    q = kernel.params["process_q"]
    t_c = log["t"][i_max]
    t0, t1 = max(0.0, t_c - 0.4), min(log["t"][-1], t_c + 0.4)
    f = np.geomspace(40, FS * 0.45, 500)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.semilogx(f, model_response_db(g, fp, q, f, FS), color="C0",
                label=f"滤波器模型（t={t_c:.2f}s, {g:.1f}dB @ {fp:.0f}Hz）")
    pts_f, pts_g = [], []
    for fm in MEAS_FREQS:
        i0, i1 = int(t0 * FS), int(t1 * FS)
        if i1 - i0 < FS // 4:
            continue
        seg = x[i0:i1]
        tt = np.arange(i0, i1) / FS
        amp = 2.0 * np.abs(np.sum(seg * np.exp(-2j * np.pi * fm * tt))) / (i1 - i0)
        if amp > 1e-4:                      # 干声里确实存在该频率才测
            pts_f.append(fm)
            pts_g.append(pilot_gain_db(x, y, fm, FS, t0, t1))
    if pts_f:
        ax.semilogx(pts_f, pts_g, "o", color="C3", ms=7,
                    label="实测（正交投影，同窗）")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("频率 Hz")
    ax.set_ylabel("增益 dB")
    ax.set_title("非目标频率改变 vs 实际滤波器模型（无响度归一化）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _run(stim_name, upload, *values):
    specs = _schema()
    params = {s.name: v for s, v in zip(specs, values)}
    x, src = _load_stimulus(stim_name, upload)
    if x.size < FS // 4:
        raise gr.Error("音频太短（<0.25s），无法处理。")

    kernel = DynamicEQ(FS)
    feedback = kernel.set_params(**params)
    y, taps = kernel.process(x)

    fb = {"音源": src,
          "参数反馈": [{"参数": n, "状态": st, "说明": m}
                     for n, st, m in feedback if st != "ok"] or "全部接受",
          "跟踪器事件": [{"t": round(e["t"], 3), "类型": e["type"]}
                      for e in kernel.events if e["type"] != "tracker_reset"] or "无",
          "极点安全钳位次数": kernel.pole_clamp_count,
          "应用参数": {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in kernel.params.items()}}

    def audio(sig):
        peak = float(np.max(np.abs(sig))) if sig.size else 0.0
        if peak >= 1.0:
            sig = sig / peak * 0.999
        return FS, sig.astype(np.float32)

    return (audio(x), audio(y), audio(taps["detect"]),
            _gain_figure(taps, kernel), _response_figure(x, y, taps, kernel),
            json.dumps(fb, ensure_ascii=False, indent=2))


def build_app():
    specs = _schema()
    with gr.Blocks(title="动态均衡母带试听面板") as app:
        gr.Markdown("## 动态均衡母带试听面板\n"
                    "控件由动态均衡内核 schema 生成；越界/非法提交由内核钳位或拒绝，"
                    "物理原因反馈在下方“内核反馈”区。")
        with gr.Row():
            with gr.Column(scale=1):
                stim = gr.Dropdown(choices=list(STIM_NAMES.keys()), value="glide",
                                   label="内置测试信号")
                upload = gr.Audio(label="或上传音频（自动重采样到 48kHz，取前 12s）",
                                  type="numpy")
                inputs = []
                for s in specs:
                    if s.kind == "bool":
                        c = gr.Checkbox(value=bool(s.default), label=s.name)
                    else:
                        c = gr.Number(value=s.default,
                                      label=f"{s.name} [{s.lo:g}, {s.hi:g}] {s.unit}")
                    inputs.append(c)
                run = gr.Button("运行动态均衡", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    dry = gr.Audio(label="干声（输入）")
                    wet = gr.Audio(label="输出（动态均衡后）")
                    det = gr.Audio(label="检测通路（带通后）")
                gain_plot = gr.Plot(label="动态增益")
                resp_plot = gr.Plot(label="响应 vs 模型")
                fb = gr.Code(label="内核反馈（钳位/拒绝/跟踪事件）", language="json")
        run.click(_run, inputs=[stim, upload, *inputs],
                  outputs=[dry, wet, det, gain_plot, resp_plot, fb])
    return app


def main():
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)


if __name__ == "__main__":
    main()
