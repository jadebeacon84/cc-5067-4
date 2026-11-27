"""母带试听面板：由动态均衡内核驱动的 Gradio 界面。

启动（在仓库根目录，二选一）:
    /tmp/venv/bin/python -I scripts/panel.py
    /tmp/venv/bin/python -m dyneq.app
然后浏览器打开终端里打印的地址（默认 http://127.0.0.1:7860）。
注意 `-I` 隐含 `-P`，不会把当前目录加入 sys.path，因此隔离模式请用
scripts/panel.py 启动器（它会先把仓库根插入 sys.path）。

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


def _audio_value(sig):
    sig = np.asarray(sig, dtype=np.float64)
    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    if peak >= 1.0:
        sig = sig / peak * 0.999
    return FS, sig.astype(np.float32)


def _ab_figure(t, g0, g1, f0, f1, fb0, fb1, req0, req1):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.8), sharex=True)
    ax = axes[0]
    ax.plot(t, g0, lw=1.4, color="C0",
            label=f"变更前动态增益（请求检测 {req0:g}Hz / 实用 {fb0:.0f}Hz）")
    ax.plot(t, g1, lw=1.4, color="C3",
            label=f"变更后动态增益（请求检测 {req1:g}Hz / 实用 {fb1:.0f}Hz）")
    ax.set_ylabel("增益 dB")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("实际修改检测频段：同一声音、同一真实 DynamicEQ 的前后对照")
    ax = axes[1]
    ax.plot(t, f0, color="C0", lw=1.2, label="变更前检测中心")
    ax.plot(t, f1, color="C3", ls="--", lw=1.2, label="变更后检测中心")
    ax.set_ylabel("检测频率 Hz")
    ax.set_xlabel("时间 s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    ax.set_title("两条检测频率轨迹（证明检测频段确实改变，非固定数据）")
    fig.tight_layout()
    return fig


def _ab_run_one(x, detect_freq, detect_q, threshold_db, follow, process_freq):
    """单次真实 DynamicEQ 运行：除检测频段外的公共参数在此固定。"""
    k = DynamicEQ(FS)
    params = dict(detect_freq=float(detect_freq),
                  detect_q=float(detect_q),
                  threshold_db=float(threshold_db),
                  ratio=4.0, max_cut_db=15.0,
                  attack_ms=5.0, release_ms=120.0, process_q=3.0)
    if follow:
        params["process_follow"] = 1
    else:
        params.update(process_follow=0, process_freq=float(process_freq))
    feedback = k.set_params(**params)
    y, taps = k.process(x)
    return k, y, taps, feedback


def _ab_compare(stim_name, upload, f_before, f_after,
                detect_q, threshold_db, follow, process_freq):
    """运行两个仅检测频段不同的真实内核，返回图/干声/A湿声/B湿声/差/JSON。"""
    x, src = _load_stimulus(stim_name, upload)
    if x.size < FS // 4:
        raise gr.Error("音频太短（<0.25s），无法做 A/B 对照。")
    try:
        req0, req1 = float(f_before), float(f_after)
    except (TypeError, ValueError):
        raise gr.Error("检测频率必须是数字（Hz）。")

    k0, y0, t0, fb0 = _ab_run_one(
        x, req0, detect_q, threshold_db, follow, process_freq)
    k1, y1, t1, fb1 = _ab_run_one(
        x, req1, detect_q, threshold_db, follow, process_freq)

    n = min(y0.size, y1.size)
    # 绘图用块率 log（t 与 log["gain_db"]/f_detect 等长）；
    # taps["gain_db"] 是逐样本率，留给指标统计。
    lg0, lg1 = t0["log"], t1["log"]
    tt = lg0["t"]
    g0, g1 = lg0["gain_db"], lg1["gain_db"]
    fd0, fd1 = lg0["f_detect"], lg1["f_detect"]
    d = y1[:n] - y0[:n]

    def _summ(k, y, taps, feedback):
        g = taps["gain_db"]
        return {
            "applied_detect_freq_hz": float(k.params["detect_freq"]),
            "applied_detect_q": float(k.params["detect_q"]),
            "applied_process_freq_hz": float(k.params["process_freq"]),
            "process_follow": int(k.params.get("process_follow", 0)),
            "gain_min_db": float(np.min(g)),
            "gain_mean_db": float(np.mean(g)),
            "samples_cut_gt1db": int(np.sum(g < -1.0)),
            "wet_rms": float(np.sqrt(np.mean(y ** 2))),
            "wet_peak": float(np.max(np.abs(y))),
            "pole_clamp_count": int(k.pole_clamp_count),
            "feedback_non_ok": [
                {"param": nm, "status": st}
                for nm, st, _msg in feedback if st != "ok"],
        }

    max_diff = float(np.max(np.abs(d))) if d.size else 0.0
    report = {
        "音源": src,
        "说明": "两次均为 dyneq.kernel.DynamicEQ 真实处理；"
                "仅 detect_freq 不同，其余公共参数一致。",
        "common": {"detect_q": float(detect_q),
                   "threshold_db": float(threshold_db),
                   "ratio": 4.0, "max_cut_db": 15.0,
                   "attack_ms": 5.0, "release_ms": 120.0,
                   "process_q": 3.0,
                   "process_follow": bool(follow),
                   "fixed_process_freq_hz": float(process_freq)},
        "before": _summ(k0, y0, t0, fb0),
        "after": _summ(k1, y1, t1, fb1),
        "output_difference_after_minus_before": {
            "max_abs": max_diff,
            "rms": float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0,
            "bit_exact_equal": bool(max_diff == 0.0),
            "audible_change": bool(max_diff > 1e-4),
        },
    }
    fig = _ab_figure(tt, g0, g1, fd0, fd1,
                     float(k0.params["detect_freq"]),
                     float(k1.params["detect_freq"]), req0, req1)
    return (fig, _audio_value(x), _audio_value(y0), _audio_value(y1),
            _audio_value(d), json.dumps(report, ensure_ascii=False, indent=2))


def _ab_preset(target_hz):
    """快捷频带按钮：把“变更后”频带设为 target_hz 并立即调用同一真实对照。"""
    def _fn(stim_name, upload, f_before, detect_q, threshold_db,
            follow, process_freq):
        outs = _ab_compare(stim_name, upload, f_before, float(target_hz),
                           detect_q, threshold_db, follow, process_freq)
        return (gr.update(value=float(target_hz)), *outs)
    return _fn


def build_app():
    specs = _schema()
    with gr.Blocks(title="动态均衡母带试听面板") as app:
        gr.Markdown("## 动态均衡母带试听面板\n"
                    "所有声音与曲线都由真实 `dyneq.kernel.DynamicEQ` 实时计算，"
                    "无固定/模拟数据。控件由内核 schema 生成；越界或非法提交由内核"
                    "钳位或拒绝，物理原因反馈在“内核反馈”区。")
        with gr.Tabs():
            # ------------------------------------------------- Tab 1 全参数运行
            with gr.Tab("完整参数运行"):
                with gr.Row():
                    with gr.Column(scale=1):
                        stim = gr.Dropdown(choices=list(STIM_NAMES.keys()),
                                           value="glide", label="内置测试信号")
                        upload = gr.Audio(
                            label="或上传音频（自动重采样到 48kHz，取前 12s）",
                            type="numpy")
                        inputs = []
                        for s in specs:
                            if s.kind == "bool":
                                c = gr.Checkbox(value=bool(s.default),
                                                 label=s.name)
                            else:
                                c = gr.Number(
                                    value=s.default,
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
                        fb = gr.Code(label="内核反馈（钳位/拒绝/跟踪事件）",
                                     language="json")
                run.click(_run, inputs=[stim, upload, *inputs],
                          outputs=[dry, wet, det, gain_plot, resp_plot, fb])

            # ------------------------------------------------- Tab 2 检测频段 A/B
            with gr.Tab("检测频段 A/B 对照"):
                gr.Markdown(
                    "### 实际修改检测频段并试听前后差异\n"
                    "对同一声音连续运行两个**真实** `DynamicEQ`：仅 `detect_freq` "
                    "不同。下图叠加两次的动态增益与检测频率轨迹；可试听干声、"
                    "变更前/后湿声以及逐样本差（后−前，未做响度归一化）。")
                with gr.Row():
                    ab_stim = gr.Dropdown(choices=list(STIM_NAMES.keys()),
                                           value="pilot_grid",
                                           label="内置测试信号")
                    ab_upload = gr.Audio(
                        label="或上传音频（重采样到 48kHz，取前 12s）",
                        type="numpy")
                with gr.Row():
                    ab_f0 = gr.Number(value=2400.0, label="变更前 detect_freq Hz")
                    ab_f1 = gr.Number(value=3600.0, label="变更后 detect_freq Hz")
                    ab_q = gr.Number(value=8.0, label="检测 Q（detect_q）")
                    ab_thr = gr.Number(value=-32.0, label="阈值 threshold_db")
                with gr.Row():
                    ab_follow = gr.Checkbox(
                        value=True,
                        label="处理频段跟随检测（process_follow；勾选后下面固定处理频段无效）")
                    ab_proc = gr.Number(value=3000.0,
                                        label="固定处理频段 process_freq Hz")
                with gr.Row():
                    ab_run = gr.Button("运行 A/B 对照", variant="primary")
                    gr.Markdown("快捷把“变更后”检测频段设为：")
                    b800 = gr.Button("800 Hz")
                    b2000 = gr.Button("2000 Hz")
                    b3000 = gr.Button("3000 Hz")
                    b6000 = gr.Button("6000 Hz")
                ab_plot = gr.Plot(label="检测频段 A/B 动态增益与频率轨迹")
                with gr.Row():
                    ab_dry = gr.Audio(label="干声")
                    ab_a = gr.Audio(label="变更前湿声 A")
                    ab_b = gr.Audio(label="变更后湿声 B")
                    ab_diff = gr.Audio(label="逐样本差 B−A（原始刻度）")
                ab_json = gr.Code(label="A/B 实测指标（真实运行结果）",
                                  language="json")

                ab_inputs = [ab_stim, ab_upload, ab_f0, ab_f1, ab_q, ab_thr,
                             ab_follow, ab_proc]
                ab_outputs = [ab_plot, ab_dry, ab_a, ab_b, ab_diff, ab_json]
                ab_run.click(_ab_compare, inputs=ab_inputs, outputs=ab_outputs)

                preset_inputs = [ab_stim, ab_upload, ab_f0, ab_q, ab_thr,
                                 ab_follow, ab_proc]
                preset_outputs = [ab_f1, *ab_outputs]
                for btn, hz in ((b800, 800), (b2000, 2000),
                                (b3000, 3000), (b6000, 6000)):
                    btn.click(_ab_preset(hz), inputs=preset_inputs,
                              outputs=preset_outputs)
        gr.Markdown(
            "命令行辅助：`python scripts/run_all.py` 生成旧样例 P1–P7；"
            "`python scripts/bench_detect_speed.py` 生成检测提速证据 "
            "out/bench/detect_speed.json 与 P8/P9。")
    return app


def main():
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)


if __name__ == "__main__":
    main()
