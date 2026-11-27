"""检测计算提速测试：缓存等价性、轨迹交叉、别名复用、共享失效、短事件。

运行: /tmp/venv/bin/python -I -m pytest tests/test_detect_speedup.py -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dyneq import kernel as opt
from dyneq import ref_baseline
from dyneq import stimuli
from dyneq.speedup import (
    LAB_BLOCK, LabNode, VersionCoeffCache, band_block_levels,
    count_detected_bursts, first_bad_segment, make_crossing_trajectories,
    overlap_sharing_scenario, run_nodes, short_burst_train,
)

FS = 48000


# ------------------------------------------------- 内核缓存：与冻结基线逐位一致
@pytest.mark.parametrize("tracker", [False, True])
def test_kernel_cache_bitexact_vs_frozen_baseline(tracker):
    params = (dict(detect_freq=2000, tracker_enable=1, process_follow=1,
                   threshold_db=-18, ratio=4, max_cut_db=15)
              if tracker else
              dict(detect_freq=3000, detect_q=4, process_freq=3000, process_q=3,
                   threshold_db=-30, ratio=3, max_cut_db=18))
    x = stimuli.pilot_grid(FS, dur=3.0)[0]
    k0 = ref_baseline.DynamicEQ(FS)
    k0.set_params(**params)
    y0, t0 = k0.process(x)
    k1 = opt.DynamicEQ(FS)
    k1.set_params(**params)
    y1, t1 = k1.process(x)
    assert np.array_equal(y0, y1)
    assert np.array_equal(t0["gain_db"], t1["gain_db"])
    assert np.array_equal(t0["log"]["f_detect"], t1["log"]["f_detect"])


def test_kernel_chunked_automation_bitexact():
    rng = np.random.default_rng(0)
    x = 0.2 * rng.standard_normal(FS * 2)
    outs = {}
    for tag, mod in (("base", ref_baseline), ("opt", opt)):
        k = mod.DynamicEQ(FS)
        k.set_params(threshold_db=-24, ratio=4, max_cut_db=12)
        ys = []
        for i in range(20):
            k.set_params(detect_freq=800 + 2000 * i / 19,
                         process_freq=900 + 1800 * i / 19)
            y, _ = k.process(x[i * 2400:(i + 1) * 2400])
            ys.append(y)
        outs[tag] = np.concatenate(ys)
    assert np.array_equal(outs["base"], outs["opt"])


def test_coeff_cache_reuses_when_params_stable():
    k = opt.DynamicEQ(FS)
    k.set_params(detect_freq=3000, process_freq=3000, threshold_db=-60)
    x = np.zeros(32 * 200)
    x[::7] = 0.01
    k.process(x)
    st = k.coeff_cache.stats()
    assert st["hits"] > 0
    assert st["misses"] > 0
    # 相同签名第二次取必然命中
    before = k.coeff_cache.hits
    a = k._cached_bandpass(1234.0, 2.0)
    b = k._cached_bandpass(1234.0, 2.0)
    assert k.coeff_cache.hits == before + 1
    assert np.array_equal(a[0], b[0])


# ------------------------------------------------- 轨迹交叉与两种缓存键
@pytest.fixture(scope="module")
def crossing_runs():
    fA, fB, info = make_crossing_trajectories(FS)
    n = int(0.5 * FS)
    tt = np.arange(n) / FS
    x = 0.4 * np.sin(2 * np.pi * info["f_sig"] * tt)
    mk = lambda tr, nm: LabNode(
        name=nm, trajectory=tr, threshold_db=-36, ratio=10, max_cut_db=20,
        q_det=12, det_attack_ms=0.1, det_release_ms=25,
        attack_ms=0.2, release_ms=12)
    nodes = [mk(fA, "A"), mk(fB, "B")]
    live = run_nodes(nodes, x, FS, "live")
    ver = run_nodes(nodes, x, FS, "version")
    idx = run_nodes(nodes, x, FS, "index", block=info["block"])
    hold = run_nodes(nodes, x, FS, "hold", block=info["block"])
    return info, tt, live, ver, idx, hold


def test_version_cache_bitexact_to_per_sample(crossing_runs):
    _, _, live, ver, _, _ = crossing_runs
    assert np.array_equal(live["y"], ver["y"])
    assert np.array_equal(live["gain_db"], ver["gain_db"])


def test_crossing_is_exactly_block_center(crossing_runs):
    info, *_ = crossing_runs
    s = info["cross_sample"]
    assert s % info["block"] == info["block"] // 2
    # 两条轨迹在交点处频率相等
    fA, fB, _ = make_crossing_trajectories(FS)
    assert abs(fA(info["t_cross"]) - fB(info["t_cross"])) < 1e-9
    # 块首确实分离，块末互换
    assert fA(info["t_cross"] - info["block"] / 2 / FS) == pytest.approx(
        info["f_at_block_start"][0], abs=1e-6)


def test_index_cache_first_bad_at_crossing(crossing_runs):
    """按频带编号的缓存必须在交点之后产生可定位的第一段错误衰减。"""
    info, tt, _, ver, idx, hold = crossing_runs
    tc = info["t_cross"]
    # B 节点：编号在交叉后易主，拿到 A 的块首系数 -> 与逐样本版本出现超差段
    err = np.abs(idx["gain_db"][1] - ver["gain_db"][1])
    seg = first_bad_segment(tt, err, tc, tol_db=0.5)
    assert seg is not None
    assert abs(seg["t0"] - tc) < 2e-4          # 首段起点紧贴交点
    assert seg["peak_err_db"] >= 1.0
    # 与稳定节点键的块保持路径相比，编号键的错配误差应显著
    err_h = np.abs(idx["gain_db"][1] - hold["gain_db"][1])
    seg_h = first_bad_segment(tt, err_h, tc, tol_db=2.0)
    assert seg_h is not None
    assert seg_h["peak_err_db"] >= 3.0
    assert idx["index_cache"]["max_freq_mismatch_hz"] > 200.0


def test_alias_display_name_only_reuses_entry():
    fA, _, _ = make_crossing_trajectories(FS)
    n = int(0.2 * FS)
    x = 0.3 * np.sin(2 * np.pi * 2700 * np.arange(n) / FS)
    a = LabNode(name="orig", trajectory=fA, traj_version="curve-xyz")
    b = LabNode(name="完全不同的显示名", trajectory=fA,
                traj_version="curve-xyz", display_name_only="仅显示名不同")
    vc = VersionCoeffCache()
    ra = run_nodes([a], x, FS, "version", version_cache=vc)
    misses_after_a = vc.misses
    h0 = vc.hits
    rb = run_nodes([b], x, FS, "version", version_cache=vc)
    assert vc.misses == misses_after_a          # 没有新系数计算
    assert vc.hits > h0
    assert np.array_equal(ra["y"], rb["y"])     # 湿声逐位相同


# ------------------------------------------------- 重叠共享 + 频移失效
def test_shared_detection_bitexact_and_invalidation():
    r = overlap_sharing_scenario(FS, dur=0.6)
    assert r["max_det_diff"] == 0.0
    assert r["max_gain_diff"] == 0.0
    assert r["wet_bitexact"]
    st = r["bank_stats"]
    # 前半段同键共享：总 lfilter 次数明显少于 2 请求/块
    assert st["cache_hits"] > 0
    assert st["lfilter_runs"] < st["block_requests"]
    assert st["unique_entries"] == 2             # 频移后多出一个键


# ------------------------------------------------- 短事件：不得靠降检测率提速
def test_polled_detection_misses_short_bursts_but_block_does_not():
    x, starts, L = short_burst_train(FS)
    lv1, t1 = band_block_levels(x, FS, poll_blocks=1)
    assert count_detected_bursts(lv1, t1, starts, FS, 0.05) == len(starts)
    lv8, t8 = band_block_levels(x, FS, poll_blocks=8)
    missed = len(starts) - count_detected_bursts(lv8, t8, starts, FS, 0.05)
    assert missed >= 5                           # 降采样检测确实漏掉短事件


def test_optimized_kernel_does_not_miss_short_events():
    x, starts, L = short_burst_train(FS)
    k = opt.DynamicEQ(FS)
    k.set_params(detect_freq=3000, detect_q=10, threshold_db=-40,
                  det_attack_ms=0.1, attack_ms=0.1, release_ms=20)
    _, taps = k.process(x)
    env = taps["env"]
    hits = sum(1 for s0 in starts if env[s0:s0 + L + LAB_BLOCK].max() > 0.05)
    assert hits >= len(starts) - 1               # 逐样本包络 + 逐块带通基本全捕获


def test_baseline_optimized_gain_output_same_on_short_events():
    x, _, _ = short_burst_train(FS, dur=1.0)
    p = dict(detect_freq=3000, detect_q=10, threshold_db=-40,
             det_attack_ms=0.1, attack_ms=0.1, release_ms=20)
    k0 = ref_baseline.DynamicEQ(FS)
    k0.set_params(**p)
    y0, t0 = k0.process(x)
    k1 = opt.DynamicEQ(FS)
    k1.set_params(**p)
    y1, t1 = k1.process(x)
    assert np.array_equal(y0, y1)
    assert np.array_equal(t0["gain_db"], t1["gain_db"])
