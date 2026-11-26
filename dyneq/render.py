"""渲染与测量：单遍渲染、WAV 导出、导频电平测量、时间常数测量。"""
from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from .kernel import DynamicEQ


def render(x, fs=48000, **params):
    """新建内核、设置参数、单遍渲染。返回 (y, taps, kernel)。"""
    k = DynamicEQ(fs)
    k.set_params(**params)
    y, taps = k.process(np.asarray(x, dtype=np.float64))
    return y, taps, k


def save_wav(path, x, fs=48000):
    """按实际电平写出，不做任何响度归一化。"""
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak >= 1.0:                       # 仅防削波，绝不提升电平
        x = x / peak * 0.999
    wavfile.write(str(path), int(fs), x.astype(np.float32))


def pilot_gain_db(dry, wet, f, fs, t0, t1):
    """在 [t0,t1] 窗内对频率 f 做正交投影，返回 wet/dry 电平差（dB）。"""
    i0, i1 = int(t0 * fs), int(t1 * fs)
    t = np.arange(i0, i1) / fs
    c = np.exp(-2j * np.pi * f * t)
    a_dry = 2.0 * np.abs(np.sum(dry[i0:i1] * c)) / (i1 - i0)
    a_wet = 2.0 * np.abs(np.sum(wet[i0:i1] * c)) / (i1 - i0)
    return 20.0 * np.log10((a_wet + 1e-12) / (a_dry + 1e-12))


def measure_time_constants(t, gain_db, t_on, t_off, cut_db, baseline_db=0.0):
    """启动/释放时间：基线扣除后的 10%->90% / 90%->10% 阶跃时间。

    cut_db 为稳态衰减幅度（正数），baseline_db 为阶跃前增益。
    """
    g = np.asarray(gain_db)
    t = np.asarray(t)
    g10 = baseline_db - 0.1 * cut_db
    g90 = baseline_db - 0.9 * cut_db
    out = {"attack_ms": float("nan"), "release_ms": float("nan")}
    seg = (t >= t_on) & (t <= t_off)
    if np.any(seg) and np.min(g[seg]) <= g90:
        st, sg = t[seg], g[seg]
        i10 = np.argmax(sg <= g10)
        i90 = np.argmax(sg <= g90)
        if sg[i10] <= g10 and sg[i90] <= g90 and i90 > i10:
            out["attack_ms"] = (st[i90] - st[i10]) * 1000.0
    seg2 = (t >= t_off) & (t <= t_off + 2.0)
    if np.any(seg2):
        st, sg = t[seg2], g[seg2]
        i90 = np.argmax(sg >= g90)
        i10 = np.argmax(sg >= g10)
        if i90 > 0 and i10 > i90:
            out["release_ms"] = (st[i10] - st[i90]) * 1000.0
    return out
