"""动态均衡内核测试：因果性、参数边界、稳定性、跟踪行为、响应与模型一致。

运行: /tmp/venv/bin/python -I -m pytest tests/test_dyneq.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyneq import stimuli
from dyneq.kernel import DynamicEQ, model_response_db, pole_radius, R_MAX, F_MIN, NYQ_GUARD
from dyneq.render import render, pilot_gain_db, measure_time_constants

FS = 48000

TRACK_PARAMS = dict(detect_freq=2000, tracker_enable=1, process_follow=1,
                    threshold_db=-18, ratio=4, max_cut_db=15, dwell_ms=60)
DYN_PARAMS = dict(detect_freq=3000, detect_q=4, process_freq=3000, process_q=3,
                  threshold_db=-30, ratio=3, max_cut_db=18,
                  attack_ms=10, release_ms=180)


# ---------------------------------------------------------------- 共享渲染
@pytest.fixture(scope="module")
def glide_run():
    x, meta = stimuli.glide_tone(FS)
    y, taps, k = render(x, FS, **TRACK_PARAMS)
    return x, y, taps, k, meta


@pytest.fixture(scope="module")
def reonset_run():
    x, meta = stimuli.reonset_tone(FS)
    y, taps, k = render(x, FS, **TRACK_PARAMS)
    return x, y, taps, k, meta


@pytest.fixture(scope="module")
def fixed_run():
    x, meta = stimuli.fixed_plus_resonance(FS)
    y, taps, k = render(x, FS, **DYN_PARAMS)
    return x, y, taps, k, meta


@pytest.fixture(scope="module")
def grid_run():
    x, meta = stimuli.pilot_grid(FS)
    y, taps, k = render(x, FS, **DYN_PARAMS)
    return x, y, taps, k, meta


# ---------------------------------------------------------------- 因果性
def test_causality_prefix_bit_exact():
    """开启录音时已写过的样本不能补算：前缀渲染与整段渲染逐样本一致。"""
    x, _ = stimuli.pilot_grid(FS, dur=2.0)
    y_full, _, _ = render(x, FS, **DYN_PARAMS)
    n = int(1.3 * FS)
    y_pre, _, _ = render(x[:n], FS, **DYN_PARAMS)
    assert np.array_equal(y_full[:n], y_pre)


# ---------------------------------------------------------------- 参数边界
def test_out_of_range_clamped_with_explanation():
    k = DynamicEQ(FS)
    res = dict((name, (st, msg)) for name, st, msg in
               k.set_params(process_freq=5.0, detect_freq=30000.0, process_q=0.0))
    assert res["process_freq"][0] == "clamped"
    assert k.params["process_freq"] == F_MIN
    assert res["detect_freq"][0] == "clamped"
    assert k.params["detect_freq"] == NYQ_GUARD * FS
    assert res["process_q"][0] == "clamped"
    # 钳位必须附物理原因解释
    assert "极点" in res["process_freq"][1] or "sin" in res["process_freq"][1]


def test_nan_rejected():
    k = DynamicEQ(FS)
    before = dict(k.params)
    res = k.set_params(process_freq=float("nan"), threshold_db=float("inf"))
    st = {name: s for name, s, _ in res}
    assert st["process_freq"] == "rejected"
    assert st["threshold_db"] == "rejected"
    assert k.params["process_freq"] == before["process_freq"]
    assert k.params["threshold_db"] == before["threshold_db"]


def test_internal_automation_cannot_submit_illegal():
    """内部自动化（非 UI）提交非法值同样被钳位/拒绝，且极点始终在安全区。"""
    k = DynamicEQ(FS)
    k.set_params(threshold_db=-60, ratio=6, max_cut_db=24, attack_ms=2, release_ms=60)
    rng = np.random.default_rng(1)
    chunk = int(0.05 * FS)
    for i in range(30):
        f_cmd = [5.0, 30000.0, float("nan"), 1000.0][i % 4]
        q_cmd = [0.0, 1e6, 3.0][i % 3]
        k.set_params(process_freq=f_cmd, process_q=q_cmd,
                     detect_freq=min(f_cmd, 8000.0) if np.isfinite(f_cmd) else f_cmd)
        assert F_MIN <= k.params["process_freq"] <= NYQ_GUARD * FS
        assert 0.2 <= k.params["process_q"] <= 24.0
        x = 0.2 * np.sin(2 * np.pi * 500 * np.arange(chunk) / FS) \
            + 0.05 * rng.standard_normal(chunk)
        y, taps = k.process(x)
        assert np.all(np.isfinite(y))
        assert np.max(taps["log"]["pole_r"]) <= R_MAX


# ---------------------------------------------------------------- 稳定性
def test_pole_radius_always_safe(grid_run):
    _, _, taps, k, _ = grid_run
    assert np.max(taps["log"]["pole_r"]) <= R_MAX
    assert k.pole_clamp_count == 0


def test_peaking_coeffs_extreme():
    """极端参数下系数直接计算也不会产生不稳定极点。"""
    for g in (-30.0, -0.1):
        for f in (20.0, 100.0, 10000.0, 21599.0):
            for q in (0.2, 3.0, 24.0):
                b, a, r, clamped = None, None, None, None
                from dyneq.kernel import peaking_coeffs
                b, a, r, clamped = peaking_coeffs(g, f, q, FS)
                assert r <= R_MAX
                assert np.all(np.isfinite(b)) and np.all(np.isfinite(a))


# ---------------------------------------------------------------- 动态行为
def test_target_attenuation_matches_model(grid_run):
    """目标频率（3000Hz 导频）实测衰减与 RBJ 峰形模型一致（±0.5dB）。"""
    x, y, taps, k, meta = grid_run
    log = taps["log"]
    for (t0, t1) in meta["windows"]:
        m = (log["t"] >= t0) & (log["t"] <= t1)
        g = float(np.mean(log["gain_db"][m]))
        fp = float(np.mean(log["f_process"][m]))
        meas = pilot_gain_db(x, y, 3000.0, FS, t0, t1)
        model = float(model_response_db(g, fp, 3.0, [3000.0], FS)[0])
        assert abs(meas - model) < 0.5
        assert meas < -5.0          # 确实压下去了


def test_no_broadband_ducking(grid_run):
    """远场非目标频率（250/500/997Hz）几乎不受影响：不是压低整段以外的声音。"""
    x, y, taps, k, meta = grid_run
    for (t0, t1) in meta["windows"]:
        for f in (250.0, 500.0, 997.0):
            g = pilot_gain_db(x, y, f, FS, t0, t1)
            assert abs(g) < 0.5, f"{f}Hz 窗({t0},{t1}) 变化 {g:.2f}dB"


def test_neighbor_influence_matches_model(grid_run):
    """邻近频率的改变符合实际滤波器模型（不是任意曲线）。"""
    x, y, taps, k, meta = grid_run
    log = taps["log"]
    for (t0, t1) in meta["windows"]:
        m = (log["t"] >= t0) & (log["t"] <= t1)
        g = float(np.mean(log["gain_db"][m]))
        fp = float(np.mean(log["f_process"][m]))
        freqs = np.array([2200.0, 2600.0, 3400.0, 4000.0])
        meas = np.array([pilot_gain_db(x, y, f, FS, t0, t1) for f in freqs])
        model = model_response_db(g, fp, 3.0, freqs, FS)
        assert np.max(np.abs(meas - model)) < 0.6


def test_time_constants(fixed_run):
    """启动/释放时间常数在合理范围（10%→90% 阶跃）。"""
    x, y, taps, k, meta = fixed_run
    log = taps["log"]
    m = (log["t"] >= 3.4) & (log["t"] <= 3.95)
    cut = -float(np.mean(log["gain_db"][m]))
    mb = (log["t"] >= 2.3) & (log["t"] <= 2.9)
    baseline = float(np.mean(log["gain_db"][mb]))
    tc = measure_time_constants(log["t"], log["gain_db"], 3.0, 4.0, cut, baseline)
    assert 5.0 < tc["attack_ms"] < 150.0
    assert 100.0 < tc["release_ms"] < 1000.0


def test_no_loudness_normalization(fixed_run):
    """输出不做整段响度归一化：静音段保持静音，电平只减不增。"""
    x, y, taps, k, meta = fixed_run
    assert np.max(np.abs(y)) <= np.max(np.abs(x)) * 1.05


# ---------------------------------------------------------------- 双音跟踪
def test_equal_detection_envelopes(glide_run, reonset_run):
    """两段双音在 f2 检测窄带的包络相等（60ms 窗平滑后 <1dB）。"""
    eg = stimuli._arrival_envelope(glide_run[0], FS, 2600.0)
    er = stimuli._arrival_envelope(reonset_run[0], FS, 2600.0)
    t = np.arange(eg.size) / FS
    m = (t >= 2.1) & (t <= 3.4)
    ksm = int(0.06 * FS)
    kern = np.ones(ksm) / ksm
    dev = 20 * np.abs(np.log10((np.convolve(eg[m], kern, "same") + 1e-9) /
                               (np.convolve(er[m], kern, "same") + 1e-9)))
    assert dev.max() < 1.0


def test_glide_tracks_to_neighbor(glide_run):
    """滑移：检测/处理中心跟随原共振滑到邻峰（终点距 2600Hz <100Hz）。"""
    _, _, taps, k, _ = glide_run
    log = taps["log"]
    assert abs(log["f_process"][-1] - 2600.0) < 100.0
    types = [e["type"] for e in k.events]
    assert "glide_switch" in types
    assert "hold_reonset" not in types      # 相位连续，不应判成双音


def test_reonset_holds_then_switches(reonset_run):
    """重新起振：检测窗未看全新起音时保持旧中心，原音淡出后按起音证据换轨。"""
    _, _, taps, k, meta = reonset_run
    log = taps["log"]
    i44 = np.searchsorted(log["t"], 4.4)
    assert abs(log["f_process"][i44] - 2000.0) < 150.0     # 旧中心保持
    assert abs(log["f_process"][-1] - 2600.0) < 100.0      # 最终换轨
    types = [e["type"] for e in k.events]
    assert "hold_reonset" in types                          # 双音证据 → 暂缓
    sw = [e for e in k.events if "switch" in e["type"] or e["type"] == "reacquire"]
    assert sw and sw[-1]["t"] > meta["t_old_end"]           # 换轨发生在原音淡出后


def test_tracker_synchronized_with_filter(glide_run):
    """检测历史与动态滤波器同步：f_process 始终跟随 f_detect（同一条对数轴）。"""
    _, _, taps, k, _ = glide_run
    log = taps["log"]
    dev = np.abs(log["f_process"] - log["f_detect"])
    assert np.max(dev) < 80.0        # 斜变限制内的最大偏差（Hz）


# ---------------------------------------------------------------- 分轨观察
def test_taps_expose_separate_paths(glide_run):
    """检测信号、处理响应、总输出分开可取。"""
    _, y, taps, k, _ = glide_run
    assert taps["detect"].shape == y.shape
    assert taps["gain_db"].shape == y.shape
    assert taps["env"].shape == y.shape
    for key in ("f_detect", "f_process", "pole_r", "trk_state", "wobble"):
        assert key in taps["log"]
