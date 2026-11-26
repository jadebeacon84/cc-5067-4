"""分析与绘图：跟踪行为、频响对照模型、时间常数、速度/误压权衡、稳定性。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from .kernel import model_response_db
from .render import pilot_gain_db

_FONT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "assets", "fonts", "NotoSansSC-Regular.otf")
if os.path.exists(_FONT):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams.update({"font.size": 9, "figure.dpi": 110, "axes.unicode_minus": False})


# ---------------------------------------------------------------- 指标
def ideal_freq_glide(t, f1, f2, t0, t1):
    u = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return f1 + (f2 - f1) * (0.5 - 0.5 * np.cos(np.pi * u))


def ideal_freq_reonset(t, f1, f2, t_switch):
    return np.where(t < t_switch, f1, f2)


def misduck_energy(t, gain_db, f_proc, f_ideal, tol_hz=150.0):
    """误压能量：处理中心偏离理想声部时施加的衰减对时间积分（dB·s）。"""
    dev = np.abs(f_proc - f_ideal) > tol_hz
    g = np.where(dev, np.abs(np.minimum(gain_db, 0.0)), 0.0)
    return float(np.trapezoid(g, t))


def tracking_delay(t, f_proc, f_target, t_ready, tol_hz=60.0):
    """从理想目标就绪到处理中心进入 tol 范围内的时间。"""
    hit = np.where((t >= t_ready) & (np.abs(f_proc - f_target) <= tol_hz))[0]
    if hit.size == 0:
        return float("nan")
    return float(t[hit[0]] - t_ready)


def crossing_lag(t, f_proc, f_ideal, f_mid):
    """处理中心与理想轨迹各自越过中点频率的时间差（跟踪滞后）。"""
    i_id = np.argmax(f_ideal >= f_mid)
    i_pr = np.argmax(f_proc >= f_mid)
    if i_id == 0 or i_pr == 0:
        return float("nan")
    return float(t[i_pr] - t[i_id])


# ---------------------------------------------------------------- 图
def plot_tracker(res_g, res_r, meta_g, meta_r, path):
    """滑移 vs 重新起振：包络证据、中心轨迹、增益、跟踪状态。"""
    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex="col")
    for col, (res, meta, title) in enumerate(
            ((res_g, meta_g, "滑移：原共振滑向邻峰（相位连续）"),
             (res_r, meta_r, "重新起振：原音继续 + 邻音起振"))):
        taps, k = res["taps"], res["kernel"]
        log = taps["log"]
        t = log["t"]
        f1, f2 = meta["f1"], meta["f2"]

        ax = axes[0, col]
        ax.plot(t, log["env_db"], label="检测包络", color="C0")
        ax.plot(t, log["gain_db"], label="动态增益 dB", color="C3")
        ax.axhline(k.params["threshold_db"], ls=":", c="gray", lw=0.8, label="阈值")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("dB")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.plot(t, log["f_detect"], label="检测中心", color="C0")
        ax.plot(t, log["f_process"], label="处理中心", color="C3", ls="--")
        if "t_glide" in meta:
            ideal = ideal_freq_glide(t, f1, f2, *meta["t_glide"])
        else:
            ideal = ideal_freq_reonset(t, f1, f2, meta["t_old_end"] + 0.15)
        ax.plot(t, ideal, label="理想声部轨迹", color="k", lw=1.2, alpha=0.6)
        ax.axhline(f1, ls=":", c="gray", lw=0.7)
        ax.axhline(f2, ls=":", c="gray", lw=0.7)
        ax.set_ylabel("Hz")
        ax.legend(loc="center right", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2, col]
        ax.plot(t, log["wobble"], color="C2", label="瞬时频率抖动 (双音证据)")
        ax.axhline(90, ls=":", c="gray", lw=0.8, label="双音判决线")
        ax.set_ylabel("Hz")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[3, col]
        state_names = ["idle", "挑战中", "暂缓", "保持旧中心", "换轨"]
        for ev in k.events:
            if ev["type"] == "tracker_reset":
                continue
            ax.axvline(ev["t"], color="C3" if "switch" in ev["type"] or ev["type"] == "reacquire" else "C1",
                       lw=1.0, alpha=0.8)
            ax.text(ev["t"], 4.4, ev["type"], rotation=90, fontsize=7, va="top")
        ax.plot(t, log["trk_state"], color="C4", drawstyle="steps-post")
        ax.set_yticks(range(5), state_names)
        ax.set_ylabel("跟踪状态")
        ax.set_xlabel("时间 s")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_response(freqs, meas_a, meas_b, model_f, model_a, model_b, g_a, g_b, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    f = np.geomspace(100, 16000, 600)
    ax.semilogx(f, model_a, color="C0", label=f"模型响应（窗A, 增益 {g_a:.1f} dB）")
    ax.semilogx(f, model_b, color="C2", label=f"模型响应（窗B, 增益 {g_b:.1f} dB）")
    ax.semilogx(freqs, meas_a, "o", color="C0", ms=7, label="导频实测（窗A）")
    ax.semilogx(freqs, meas_b, "s", color="C2", ms=7, label="导频实测（窗B）")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("频率 Hz")
    ax.set_ylabel("增益 dB")
    ax.set_title("实测非目标频率响应 vs 峰形滤波器模型（无宽带压低、无响度归一化）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_timeconst(t, gain_db, bursts, meas, cfg_attack, cfg_release, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, gain_db, color="C3", lw=1.2, label="动态增益")
    for (t0, t1, lvl) in bursts:
        ax.axvspan(t0, t1, color="gray", alpha=0.15)
        ax.text((t0 + t1) / 2, 0.5, f"共振 {lvl:.0f} dBFS", ha="center", fontsize=8)
    ax.annotate(f"实测启动 {meas['attack_ms']:.0f} ms（配置 {cfg_attack:.0f} ms）",
                xy=(3.05, -5), xytext=(2.1, -11),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    ax.annotate(f"实测释放 {meas['release_ms']:.0f} ms（配置 {cfg_release:.0f} ms）",
                xy=(4.3, -8), xytext=(4.35, -14),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    ax.set_xlabel("时间 s")
    ax.set_ylabel("增益 dB")
    ax.set_title("时间常数：固定音 + 可变窄带共振")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_tradeoff(dwells, delay_g, mis_g, delay_r, mis_r, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dwells, mis_g, "o-", color="C0", label="误压能量·滑移")
    ax.plot(dwells, mis_r, "s-", color="C2", label="误压能量·重新起振")
    ax.set_xlabel("驻留时间 dwell ms")
    ax.set_ylabel("误压能量 dB·s")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(dwells, delay_g, "o--", color="C0", alpha=0.6, label="跟随延迟·滑移")
    ax2.plot(dwells, delay_r, "s--", color="C2", alpha=0.6, label="换轨延迟·重新起振")
    ax2.set_ylabel("延迟 s")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.set_title("跟踪速度 vs 误压能量的权衡（检测历史与动态滤波器同步）")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_stability(t, pole_r, t_sub, f_sub, f_applied, fs, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].semilogy(t, pole_r, lw=1.0, color="C3")
    axes[0].axhline(1.0, color="k", lw=0.8, ls="--", label="稳定边界 |p|=1")
    axes[0].axhline(0.99995, color="gray", lw=0.8, ls=":", label="内核安全上限")
    axes[0].set_ylabel("极点半径")
    axes[0].set_ylim(1e-2, 2)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("连续自动化扫描下极点半径始终在安全区")
    axes[1].plot(t_sub, f_sub, "o", ms=4, label="自动化提交值（含非法值）", color="C1")
    axes[1].plot(t, f_applied, lw=1.2, label="内核实际应用值（钳位后）", color="C0")
    axes[1].axhline(20, color="gray", ls=":", lw=0.8)
    axes[1].axhline(0.45 * fs, color="gray", ls=":", lw=0.8)
    axes[1].set_ylabel("处理中心 Hz")
    axes[1].set_xlabel("时间 s")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spectrograms(pairs, fs, path):
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 3.2 * len(pairs)), sharex=True)
    for row, (name, dry, wet) in enumerate(pairs):
        for col, (sig, tag) in enumerate(((dry, "干声"), (wet, "动态均衡输出"))):
            f, tt, S = spectrogram(sig, fs, nperseg=1024, noverlap=768)
            ax = axes[row, col]
            ax.pcolormesh(tt, f / 1000, 10 * np.log10(S + 1e-14),
                          shading="auto", cmap="magma", vmin=-110, vmax=-30)
            ax.set_ylim(0, 8)
            ax.set_ylabel(f"{name}\n频率 kHz")
            ax.set_title(f"{name} — {tag}", fontsize=9)
    axes[-1, 0].set_xlabel("时间 s")
    axes[-1, 1].set_xlabel("时间 s")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_taps(t_seg, detect, env_db, gain_db, out, path):
    """检测信号、处理响应、总输出分开观察。"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_seg, detect, lw=0.6, color="C0")
    axes[0].set_ylabel("检测信号")
    axes[0].set_title("检测通路（带通后）")
    axes[1].plot(t_seg, env_db, lw=1.0, color="C2")
    axes[1].set_ylabel("包络 dB")
    axes[1].set_title("检测包络")
    axes[2].plot(t_seg, gain_db, lw=1.0, color="C3")
    axes[2].set_ylabel("增益 dB")
    axes[2].set_title("处理响应（动态增益）")
    axes[3].plot(t_seg, out, lw=0.6, color="C4")
    axes[3].set_ylabel("输出")
    axes[3].set_title("总输出")
    axes[3].set_xlabel("时间 s")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def measure_response(dry, wet, freqs, fs, windows, taps):
    """逐导频实测增益，并与同窗内平均参数下的模型响应对照。"""
    log = taps["log"]
    out = []
    for (t0, t1) in windows:
        m = (log["t"] >= t0) & (log["t"] <= t1)
        g = float(np.mean(log["gain_db"][m]))
        fp = float(np.mean(log["f_process"][m]))
        meas = np.array([pilot_gain_db(dry, wet, f, fs, t0, t1) for f in freqs])
        model = model_response_db(g, fp, 3.0, freqs, fs)
        out.append({"window": (t0, t1), "gain_db": g, "f_process": fp,
                    "meas": meas, "model": model,
                    "max_err": float(np.max(np.abs(meas - model)))})
    return out
