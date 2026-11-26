"""测试信号：双音（滑移 / 重新起振）、导频、固定音+可变窄带共振。

所有信号都叠加低电平标定导频，用于量化非目标声部被误伤的程度。
导频频率避开检测/处理频段，电平远低于阈值，不参与检测。
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from .kernel import bandpass_coeffs

PILOTS = ((500.0, 0.030), (997.0, 0.040), (3400.0, 0.025), (6000.0, 0.030))


def _add_pilots(x, fs, pilots=PILOTS):
    t = np.arange(x.size) / fs
    out = x.copy()
    for f, a in pilots:
        out += a * np.sin(2.0 * np.pi * f * t)
    return out


def glide_tone(fs=48000, f1=2000.0, f2=2600.0, t_glide=(2.0, 3.0), dur=6.0,
               amp=0.5, with_pilots=True):
    """原共振从 f1 滑向邻峰 f2（相位连续，同一声部）。"""
    n = int(dur * fs)
    t = np.arange(n) / fs
    u = np.clip((t - t_glide[0]) / (t_glide[1] - t_glide[0]), 0.0, 1.0)
    s = 0.5 - 0.5 * np.cos(np.pi * u)          # raised-cosine 频率轨迹
    f_inst = f1 + (f2 - f1) * s
    phase = 2.0 * np.pi * np.cumsum(f_inst) / fs
    x = amp * np.sin(phase)
    if with_pilots:
        x = _add_pilots(x, fs)
    return x, {"f1": f1, "f2": f2, "t_glide": t_glide, "dur": dur, "amp": amp}


def _arrival_envelope(x, fs, f2, bw=150.0):
    """信号经 f2 检测窄带（与跟踪器频带相同的复数谐振器）后的包络。"""
    r = float(np.exp(-np.pi * bw / fs))
    w = 2.0 * np.pi * f2 / fs
    y = lfilter([1.0 - r], [1.0, -r * np.exp(1j * w)], x)
    env = 2.0 * np.abs(y)          # 解析幅度 ≈ 音调振幅
    k = 1.0 - np.exp(-1.0 / (0.008 * fs))
    out = np.empty_like(env)
    e = 0.0
    for i, v in enumerate(env):
        e += k * (v - e)
        out[i] = e
    return out


def reonset_tone(fs=48000, f1=2000.0, f2=2600.0, t_onset=2.0, t_old_end=4.5,
                 dur=6.0, amp=0.5, with_pilots=True, arrival=None):
    """原音在 f1 继续，邻音 f2 重新起振（两个独立声部）。

    新音振幅经迭代标定：反复测量"原音+新音"在 f2 检测窄带的包络并乘性
    校正，使检测器看到的包络与滑移情形相等（残余偏差仅为原音裙边与新音
    差频起伏的物理极限）。检测窗尚未看全新起音时，两种情形无法区分。
    """
    n = int(dur * fs)
    t = np.arange(n) / fs
    old_full = amp * np.sin(2.0 * np.pi * f1 * t)
    if arrival is None:
        g, _ = glide_tone(fs, f1, f2, (t_onset, t_onset + 1.0), dur, amp,
                          with_pilots=False)
        target = _arrival_envelope(g, fs, f2)
        new_amp = target.copy()
        for _ in range(3):      # 迭代标定：包络乘性校正
            new = new_amp * np.sin(2.0 * np.pi * f2 * t + 0.7)
            new[t < t_onset] = 0.0
            meas = _arrival_envelope(old_full + new, fs, f2)
            corr = (target + 1e-4) / (meas + 1e-4)
            ksm = max(1, int(0.020 * fs))
            corr = np.convolve(corr, np.ones(ksm) / ksm, "same")
            new_amp = np.clip(new_amp * np.clip(corr, 0.5, 2.0), 0.0, None)
        arrival = new_amp
    fade = 0.5 - 0.5 * np.cos(np.pi * np.clip((t_old_end + 0.15 - t) / 0.15, 0.0, 1.0))
    old = old_full * fade
    new = arrival * np.sin(2.0 * np.pi * f2 * t + 0.7)
    new[t < t_onset] = 0.0
    x = old + new
    if with_pilots:
        x = _add_pilots(x, fs)
    return x, {"f1": f1, "f2": f2, "t_onset": t_onset, "t_old_end": t_old_end,
               "dur": dur, "amp": amp, "arrival": arrival}


def fixed_plus_resonance(fs=48000, f_fixed=997.0, f_res=3000.0, dur=7.0,
                         seed=7):
    """固定音 + 电平阶跃变化的窄带共振（测时间常数与目标衰减）。"""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    noise = rng.standard_normal(n)
    b, a = bandpass_coeffs(f_res, 15.0, fs)
    res = lfilter(b, a, noise)
    res /= np.sqrt(np.mean(res ** 2))          # 归一化到单位 RMS（仅对噪声源，非输出）
    gate = np.zeros(n)
    for (t0, t1, lvl_db) in ((1.0, 2.0, -22.0), (3.0, 4.0, -12.0), (5.0, 6.0, -17.0)):
        seg = (t >= t0) & (t < t1)
        gate[seg] = 10.0 ** (lvl_db / 20.0)
    x = 0.25 * np.sin(2.0 * np.pi * f_fixed * t) + gate * res
    x += 0.020 * np.sin(2.0 * np.pi * 2400.0 * t)   # 邻近导频
    x += 0.020 * np.sin(2.0 * np.pi * 3600.0 * t)
    x += 0.030 * np.sin(2.0 * np.pi * 500.0 * t)    # 远场导频
    return x, {"f_fixed": f_fixed, "f_res": f_res, "dur": dur,
               "bursts": ((1.0, 2.0, -22.0), (3.0, 4.0, -12.0), (5.0, 6.0, -17.0))}


def pilot_grid(fs=48000, f_res=3000.0, dur=8.0, seed=11):
    """导频栅格 + 触发共振：逐点测量非目标频率的实际响应，对照滤波器模型。"""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    freqs = np.array([250, 500, 997, 1500, 2200, 2600, 3000, 3400, 4000,
                      6000, 9000, 12000], dtype=float)
    x = np.zeros(n)
    for f in freqs:
        x += (10.0 ** (-40.0 / 20.0)) * np.sin(2.0 * np.pi * f * t)
    noise = rng.standard_normal(n)
    b, a = bandpass_coeffs(f_res, 15.0, fs)
    res = lfilter(b, a, noise)
    res /= np.sqrt(np.mean(res ** 2))
    gate = np.zeros(n)
    for (t0, t1, lvl_db) in ((2.0, 3.5, -12.0), (5.0, 6.5, -18.0)):
        seg = (t >= t0) & (t < t1)
        gate[seg] = 10.0 ** (lvl_db / 20.0)
    x += gate * res
    return x, {"freqs": freqs, "f_res": f_res, "dur": dur,
               "windows": ((2.5, 3.4), (5.5, 6.4))}


STIMULI = {
    "glide": glide_tone,
    "reonset": reonset_tone,
    "fixed_resonance": fixed_plus_resonance,
    "pilot_grid": pilot_grid,
}
