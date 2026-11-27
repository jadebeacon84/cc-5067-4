"""检测计算提速基准：成本分解、缓存键交叉探针、共享检测、短事件、频段响应。

本脚本所有数字均为当次真实执行得到，写入 out/bench/detect_speed.json，
音频证据写入 out/audio/speedup_*.wav，图写入 out/figs/P8/P9。
速度对照使用冻结参照 dyneq.ref_baseline（提速前 HEAD 内核逐字节副本）。

用法（仓库根目录）:
    /tmp/venv/bin/python -I scripts/bench_detect_speed.py
"""
from __future__ import annotations

import cProfile
import json
import os
import platform
import pstats
import subprocess
import sys
import io
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.signal import lfilter

from dyneq import ref_baseline, stimuli
from dyneq.kernel import DynamicEQ, R_MAX
from dyneq.render import save_wav, pilot_gain_db, measure_time_constants
from dyneq.speedup import (
    LAB_BLOCK, LabNode, SharedDetectionBank, VersionCoeffCache,
    band_block_levels, count_detected_bursts, first_bad_segment,
    make_crossing_trajectories, overlap_sharing_scenario, run_nodes,
    short_burst_train,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
AUDIO = os.path.join(OUT, "audio")
FIGS = os.path.join(OUT, "figs")
BENCH = os.path.join(OUT, "bench")
FS = 48000

DYN_PARAMS = dict(detect_freq=3000, detect_q=4, process_freq=3000, process_q=3,
                  threshold_db=-30, ratio=3, max_cut_db=18,
                  attack_ms=10, release_ms=180)
TRK_PARAMS = dict(detect_freq=2000, tracker_enable=1, process_follow=1,
                  threshold_db=-18, ratio=4, max_cut_db=15, dwell_ms=60)


def jsafe(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------- 计时
def paired_time_process(params, x, rounds=11):
    """交替执行基线/优化，按轮配对取中位数，降低调度与热漂移偏差。

    返回 (base_us, opt_us, paired_ratios, yb, tb, yo, to, ko)。
    每轮都新建内核，避免缓存/状态跨轮污染。
    """
    nblocks = int(np.ceil(x.size / DynamicEQ.BLOCK))
    rb, ro, ratios = [], [], []
    last_b = last_o = None
    for j in range(rounds):
        order = (ref_baseline.DynamicEQ, DynamicEQ) if j % 2 == 0 else \
                (DynamicEQ, ref_baseline.DynamicEQ)
        pair = {}
        for Cls in order:
            k = Cls(FS)
            k.set_params(**params)
            t0 = time.perf_counter()
            y, taps = k.process(x)
            dt = time.perf_counter() - t0
            pair[Cls] = (dt, y, taps, k)
        db, yb, tb, kb = pair[ref_baseline.DynamicEQ]
        do, yo, to, ko = pair[DynamicEQ]
        rb.append(db); ro.append(do); ratios.append(db / do)
        last_b = (yb, tb, kb)
        last_o = (yo, to, ko)
    ub = float(np.median(rb)) / nblocks * 1e6
    uo = float(np.median(ro)) / nblocks * 1e6
    return (ub, uo, [float(r) for r in ratios],
            last_b[0], last_b[1], last_o[0], last_o[1], last_o[2])


def paired_time_sweep(x, rounds=9):
    """连续调参负载的交替配对计时（与 paired_time_process 同理）。"""
    nblocks = x.size / DynamicEQ.BLOCK
    rb, ro, ratios = [], [], []
    ko_last = None
    for j in range(rounds):
        order = ("base", "opt") if j % 2 == 0 else ("opt", "base")
        pair = {}
        for tag in order:
            Cls = ref_baseline.DynamicEQ if tag == "base" else DynamicEQ
            k = Cls(FS)
            k.set_params(threshold_db=-26, ratio=5, max_cut_db=14)
            t0 = time.perf_counter()
            pos, i, chunk = 0, 0, int(0.05 * FS)
            while pos < x.size:
                fd = 800 + 2400 * (0.5 + 0.5 * np.sin(i * 0.35))
                k.set_params(detect_freq=fd, detect_q=2 + 5 * (i % 5) / 4,
                             process_freq=fd * 0.95)
                k.process(x[pos:pos + chunk])
                pos += chunk
                i += 1
            dt = time.perf_counter() - t0
            pair[tag] = (dt, k)
        db, _ = pair["base"]
        do, ko = pair["opt"]
        rb.append(db); ro.append(do); ratios.append(db / do)
        ko_last = ko
    ub = float(np.median(rb)) / nblocks * 1e6
    uo = float(np.median(ro)) / nblocks * 1e6
    return ub, uo, [float(r) for r in ratios], ko_last


def cost_decomposition(nblocks=6000):
    """cProfile 内部时间按类别聚合（跟踪器负载，最能体现系数重复构造）。"""
    rng = np.random.default_rng(0)
    x = 0.15 * rng.standard_normal(nblocks * 32)
    k = DynamicEQ(FS)
    k.set_params(**TRK_PARAMS)
    pr = cProfile.Profile()
    pr.enable()
    k.process(x)
    pr.disable()
    groups = {"lfilter/scipy": 0.0, "系数计算": 0.0,
              "tracker 判决 Python": 0.0, "process 包络/增益循环": 0.0,
              "其他": 0.0}
    coeff_funcs = {"peaking_coeffs", "bandpass_coeffs", "pole_radius"}
    st = pstats.Stats(pr)
    for (_fn, _ln, func), (_cc, _nc, tt, _ct, _callers) in st.stats.items():
        if func in ("lfilter",) or "_linear_filter" in func:
            groups["lfilter/scipy"] += tt
        elif func in coeff_funcs or func == "get":
            groups["系数计算"] += tt
        elif func == "_tracker_step":
            groups["tracker 判决 Python"] += tt
        elif func == "process":
            groups["process 包络/增益循环"] += tt
        else:
            groups["其他"] += tt
    total = sum(groups.values())
    return {"nblocks": nblocks,
            "time_ms_by_category": {k_: 1000 * v for k_, v in groups.items()},
            "pct_by_category": {k_: 100 * v / total for k_, v in groups.items()
                                if total > 0},
            "profiler_total_s": total}


# ---------------------------------------------------------------- 共享扩展计时
def sharing_scaling(block_n=3000):
    rng = np.random.default_rng(2)
    blk = 0.2 * rng.standard_normal(32)
    from dyneq.kernel import bandpass_coeffs
    b, a = bandpass_coeffs(2500.0, 6.0, FS)
    out = {}
    for N in (2, 4, 8):
        # 独立：N 份 zi，每块 N 次 lfilter
        zis = [np.zeros(2) for _ in range(N)]

        def run_indep():
            for _ in range(block_n):
                for j in range(N):
                    y, zis[j] = lfilter(b, a, blk, zi=zis[j])

        ti = min(_timeit(run_indep) for _ in range(5))
        # 全重叠共享：一块只跑一次 lfilter，其余分发缓存输出
        bank = SharedDetectionBank(FS)
        key = (2500.0, 6.0, FS, "v")

        def run_shared():
            for _ in range(block_n):
                bank.begin_block()
                for _j in range(N):
                    bank.process_block(key, 2500.0, 6.0, blk)

        ts = min(_timeit(run_shared) for _ in range(5))
        out[N] = {"independent_us_per_block": ti / block_n * 1e6,
                  "shared_us_per_block": ts / block_n * 1e6,
                  "speedup": ti / ts}
    return out


def _timeit(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(BENCH, exist_ok=True)
    os.makedirs(AUDIO, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    report = {"provenance": {
        "git_head": git_sha(), "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "started_unix": time.time()}}

    # ========== 1. 内核速度 + 与冻结基线的逐样本等价 ==========
    print("[1/7] 内核基准：冻结基线 vs 系数缓存优化 ...")
    x_grid = stimuli.pilot_grid(FS, dur=4.0)[0]
    x_glide = stimuli.glide_tone(FS)[0]
    speed = {}
    for name, params, x in [
            ("dynamic_pilot", DYN_PARAMS, x_grid),
            ("tracker_glide", TRK_PARAMS, x_glide)]:
        (ub, uo, pr, yb, tb, yo, to, ko) = paired_time_process(params, x)
        speed[name] = {
            "baseline_us_per_block": ub, "optimized_us_per_block": uo,
            "speedup_x": ub / uo,
            "paired_round_speedups": pr,
            "paired_speedup_median_x": float(np.median(pr)),
            "timing": "interleaved_paired_median",
            "output_bitexact_vs_baseline": bool(np.array_equal(yb, yo)),
            "gain_bitexact_vs_baseline": bool(
                np.array_equal(tb["gain_db"], to["gain_db"])),
            "coeff_cache": ko.coeff_cache.stats()}
        print(f"   {name}: {ub:.1f} -> {uo:.1f} us/blk ({ub/uo:.3f}x, "
              f"paired med {np.median(pr):.3f}x), "
              f"bitexact={speed[name]['output_bitexact_vs_baseline']}")
    # 连续调参负载（交替配对计时）
    x_sweep = 0.15 * np.random.default_rng(7).standard_normal(FS * 2)
    ub_s, uo_s, pr_s, k_s = paired_time_sweep(x_sweep)
    speed["continuous_param_sweep"] = {
        "baseline_us_per_block": ub_s, "optimized_us_per_block": uo_s,
        "speedup_x": ub_s / uo_s,
        "paired_round_speedups": pr_s,
        "paired_speedup_median_x": float(np.median(pr_s)),
        "timing": "interleaved_paired_median",
        "coeff_cache": k_s.coeff_cache.stats()}
    print(f"   continuous_param_sweep: {ub_s:.1f} -> {uo_s:.1f} us/blk "
          f"({ub_s/uo_s:.3f}x, paired med {np.median(pr_s):.3f}x)")
    report["kernel_speed"] = speed

    # ========== 2. 成本分解 ==========
    print("[2/7] cProfile 成本分解 ...")
    report["cost_decomposition_tracker"] = cost_decomposition()
    report["sharing_scaling"] = sharing_scaling()

    # ========== 3. 交叉轨迹缓存探针 ==========
    print("[3/7] 轨迹交叉 / 缓存键探针 ...")
    fA, fB, info = make_crossing_trajectories(FS)
    n_cross = int(0.5 * FS)
    tt = np.arange(n_cross) / FS
    x_cross = 0.4 * np.sin(2 * np.pi * info["f_sig"] * tt) \
        + 0.004 * np.random.default_rng(0).standard_normal(n_cross)
    mk = lambda tr, nm, ver="curve-v1": LabNode(
        name=nm, trajectory=tr, traj_version=ver, threshold_db=-36, ratio=10,
        max_cut_db=20, q_det=12, det_attack_ms=0.1, det_release_ms=25,
        attack_ms=0.2, release_ms=12)
    nodes = [mk(fA, "A"), mk(fB, "B")]
    r_live = run_nodes(nodes, x_cross, FS, "live")
    r_ver = run_nodes(nodes, x_cross, FS, "version")
    r_idx = run_nodes(nodes, x_cross, FS, "index", block=info["block"])
    r_hold = run_nodes(nodes, x_cross, FS, "hold", block=info["block"])
    tc = info["t_cross"]
    cross_metrics = {"scenario": info, "nodes": {}}
    for j, nm in enumerate(("A", "B")):
        ev = np.abs(r_ver["gain_db"][j] - r_live["gain_db"][j])
        ei = np.abs(r_idx["gain_db"][j] - r_ver["gain_db"][j])
        eh = np.abs(r_idx["gain_db"][j] - r_hold["gain_db"][j])
        cross_metrics["nodes"][nm] = {
            "version_vs_sample_max_err_db": float(ev.max()),
            "index_vs_version_first_bad_0p5db": first_bad_segment(tt, ei, tc, 0.5),
            "index_vs_hold_first_bad_2db": first_bad_segment(tt, eh, tc, 2.0),
            "index_vs_version_max_post_cross_err_db": float(ei[tt >= tc].max()),
            "index_vs_hold_max_post_cross_err_db": float(eh[tt >= tc].max())}
    cross_metrics["index_cache"] = r_idx["index_cache"]
    cross_metrics["version_bitexact_to_sample"] = bool(
        np.array_equal(r_ver["y"], r_live["y"]))
    report["crossing_probe"] = cross_metrics

    save_wav(os.path.join(AUDIO, "speedup_cross_wet_version.wav"), r_ver["y"], FS)
    save_wav(os.path.join(AUDIO, "speedup_cross_wet_sample_ref.wav"), r_live["y"], FS)
    save_wav(os.path.join(AUDIO, "speedup_cross_wet_index_bandno.wav"), r_idx["y"], FS)
    d_ver_sample = r_ver["y"] - r_live["y"]
    d_idx_sample = r_idx["y"] - r_live["y"]
    save_wav(os.path.join(AUDIO, "speedup_cross_diff_version_minus_sample.wav"),
             d_ver_sample, FS)
    save_wav(os.path.join(AUDIO, "speedup_cross_diff_index_minus_sample.wav"),
             d_idx_sample, FS)
    # 交点前后 80ms 精听片段
    a0, a1 = int((tc - 0.08) * FS), int((tc + 0.12) * FS)
    for tag, sig in (("version", r_ver["y"]), ("index", r_idx["y"]),
                     ("sample", r_live["y"])):
        save_wav(os.path.join(AUDIO, f"speedup_cross_clip_{tag}.wav"),
                 sig[a0:a1], FS)

    # ========== 4. 仅显示名不同的别名对照 ==========
    print("[4/7] 显示名别名复用对照 ...")
    n_al = int(0.4 * FS)
    x_al = 0.3 * np.sin(2 * np.pi * 2700 * np.arange(n_al) / FS)
    n1 = LabNode(name="node-original", trajectory=fA, traj_version="alias-v9")
    n2 = LabNode(name="界面显示名-完全不同", trajectory=fA,
                 traj_version="alias-v9", display_name_only="只改 label")
    vc = VersionCoeffCache()
    ra = run_nodes([n1], x_al, FS, "version", version_cache=vc)
    miss_after_first = vc.misses
    hit_after_first = vc.hits
    rb = run_nodes([n2], x_al, FS, "version", version_cache=vc)
    alias = {
        "new_misses_on_rename": vc.misses - miss_after_first,
        "new_hits_on_rename": vc.hits - hit_after_first,
        "wet_bitexact": bool(np.array_equal(ra["y"], rb["y"])),
        "wet_max_abs_diff": float(np.max(np.abs(ra["y"] - rb["y"])))}
    report["alias_control"] = alias
    save_wav(os.path.join(AUDIO, "speedup_alias_original.wav"), ra["y"], FS)
    save_wav(os.path.join(AUDIO, "speedup_alias_renamed.wav"), rb["y"], FS)
    save_wav(os.path.join(AUDIO, "speedup_alias_diff.wav"), ra["y"] - rb["y"], FS)

    # ========== 5. 频带重叠共享 + 频移部分失效 + 全量衰减核验 ==========
    print("[5/7] 重叠共享 / 频移失效 / 全量衰减 ...")
    sh = overlap_sharing_scenario(FS, dur=1.2)
    atten = {}
    for tag, wet in (("shared", sh["wet_shared"]), ("independent", sh["wet_indep"])):
        atten[tag] = {}
        for win, (t0, t1) in {"pre": (0.1, 0.45), "post": (0.65, 1.1)}.items():
            for f in (1800.0, 3000.0):
                atten[tag][f"{int(f)}Hz_{win}"] = pilot_gain_db(
                    sh["x"], wet, f, FS, t0, t1)
    report["overlap_sharing"] = {
        "bank_stats": sh["bank_stats"],
        "max_det_diff": sh["max_det_diff"], "max_gain_diff": sh["max_gain_diff"],
        "wet_bitexact": sh["wet_bitexact"], "attenuation_db": atten}
    save_wav(os.path.join(AUDIO, "speedup_overlap_wet_shared.wav"),
             sh["wet_shared"], FS)

    # ========== 6. 短事件 + 起释/稳定性 + 同声音增益/输出差异 ==========
    print("[6/7] 短事件、时间常数、稳定性与同声音差异 ...")
    x_burst, starts, Lb = short_burst_train(FS, dur=1.0)
    short_ev = {"n_bursts": int(len(starts)), "burst_ms": 2.0}
    for poll in (1, 4, 8, 16):
        lv, tlv = band_block_levels(x_burst, FS, poll_blocks=poll)
        short_ev[f"poll{poll}_hits"] = count_detected_bursts(
            lv, tlv, starts, FS, 0.05)
    k_b = DynamicEQ(FS)
    k_b.set_params(detect_freq=3000, detect_q=10, threshold_db=-40,
                   det_attack_ms=0.1, attack_ms=0.1, release_ms=20)
    _, tb_b = k_b.process(x_burst)
    short_ev["kernel_env_hits"] = int(sum(
        1 for s0 in starts
        if tb_b["env"][s0:s0 + Lb + LAB_BLOCK].max() > 0.05))
    # 同一短峰声音：基线 vs 优化的增益轨迹/输出差异
    kb0 = ref_baseline.DynamicEQ(FS)
    p_short = dict(detect_freq=3000, detect_q=10, threshold_db=-40,
                   det_attack_ms=0.1, attack_ms=0.1, release_ms=20)
    kb0.set_params(**p_short)
    yb0, tb0 = kb0.process(x_burst)
    kb1 = DynamicEQ(FS)
    kb1.set_params(**p_short)
    yb1, tb1 = kb1.process(x_burst)
    short_ev["baseline_vs_optimized_output_bitexact"] = bool(
        np.array_equal(yb0, yb1))
    short_ev["gain_trajectory_max_diff_db"] = float(
        np.max(np.abs(tb0["gain_db"] - tb1["gain_db"])))
    report["short_events"] = short_ev

    # 起释时间常数 + 极点稳定（旧样例 fixed_resonance）
    xf, mf = stimuli.fixed_plus_resonance(FS)
    stability = {}
    for tag, Cls in (("baseline", ref_baseline.DynamicEQ), ("optimized", DynamicEQ)):
        k = Cls(FS)
        k.set_params(**DYN_PARAMS)
        yf, tf = k.process(xf)
        logf = tf["log"]
        msk = (logf["t"] >= 3.4) & (logf["t"] <= 3.95)
        cut = -float(np.mean(logf["gain_db"][msk]))
        base_msk = (logf["t"] >= 2.3) & (logf["t"] <= 2.9)
        bl = float(np.mean(logf["gain_db"][base_msk]))
        tconst = measure_time_constants(logf["t"], logf["gain_db"], 3.0, 4.0, cut, bl)
        stability[tag] = {"attack_ms": tconst["attack_ms"],
                          "release_ms": tconst["release_ms"],
                          "pole_r_max": float(np.max(logf["pole_r"])),
                          "pole_clamp_count": k.pole_clamp_count,
                          "output_finite": bool(np.all(np.isfinite(yf)))}
    report["attack_release_stability"] = stability

    # ========== 7. 频段响应（pilot_grid 实测 vs RBJ 模型）+ 每块成本 ==========
    print("[7/7] 频段响应实测 + 出图/写 JSON ...")
    xp, mp = stimuli.pilot_grid(FS)
    kp = DynamicEQ(FS)
    kp.set_params(**DYN_PARAMS)
    t_resp0 = time.perf_counter()
    yp, tp = kp.process(xp)
    resp_us = (time.perf_counter() - t_resp0) / (xp.size / 32) * 1e6
    log = tp["log"]
    resp_windows = []
    for (t0, t1) in mp["windows"]:
        m = (log["t"] >= t0) & (log["t"] <= t1)
        g = float(np.mean(log["gain_db"][m]))
        fp = float(np.mean(log["f_process"][m]))
        freqs = mp["freqs"]
        meas = np.array([pilot_gain_db(xp, yp, f, FS, t0, t1) for f in freqs])
        from dyneq.kernel import model_response_db
        model = model_response_db(g, fp, 3.0, freqs, FS)
        resp_windows.append({"window": [t0, t1], "gain_db": g, "f_process": fp,
                             "max_model_err_db": float(np.max(np.abs(meas - model))),
                             "meas_db": meas, "model_db": model})
    report["band_response"] = {
        "windows": resp_windows, "per_block_us_this_run": resp_us,
        "coeff_cache": kp.coeff_cache.stats()}

    report["provenance"]["finished_unix"] = time.time()
    with open(os.path.join(BENCH, "detect_speed.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=jsafe)

    _plot(report, tt, tc, r_live, r_ver, r_idx, info, xp, yp, tp, mp)
    print("完成 -> out/bench/detect_speed.json, out/figs/P8/P9")
    _print_summary(report)


def _print_summary(r):
    for name, s in r["kernel_speed"].items():
        print(f"  speed {name}: {s['baseline_us_per_block']:.1f} -> "
              f"{s['optimized_us_per_block']:.1f} us/blk "
              f"({s['speedup_x']:.3f}x) bitexact={s.get('output_bitexact_vs_baseline')}")
    c = r["crossing_probe"]["nodes"]["B"]
    print("  crossing B first bad:", c["index_vs_version_first_bad_0p5db"])
    print("  alias:", r["alias_control"])
    print("  overlap:", r["overlap_sharing"]["bank_stats"],
          "bitexact=", r["overlap_sharing"]["wet_bitexact"])
    print("  short events:", {k: v for k, v in r["short_events"].items()
                              if "poll" in k or "kernel" in k})


def _plot(report, tt, tc, r_live, r_ver, r_idx, info, xp, yp, tp, mp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font = os.path.join(ROOT, "assets", "fonts", "NotoSansSC-Regular.otf")
    if os.path.exists(font):
        from matplotlib import font_manager
        font_manager.fontManager.addfont(font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    # 1 每块成本
    names = [k for k in report["kernel_speed"] if k != "continuous_param_sweep"]
    b = [report["kernel_speed"][k]["baseline_us_per_block"] for k in names]
    o = [report["kernel_speed"][k]["optimized_us_per_block"] for k in names]
    xi = np.arange(len(names))
    ax[0, 0].bar(xi - 0.2, b, 0.4, label="冻结基线")
    ax[0, 0].bar(xi + 0.2, o, 0.4, label="系数缓存优化")
    ax[0, 0].set_xticks(xi, ["动态峰形", "13 频带跟踪器"])
    ax[0, 0].set_ylabel("µs / 块（中位）")
    ax[0, 0].set_title("每块成本（真实计时）")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3, axis="y")

    # 2 成本分解
    cd = report["cost_decomposition_tracker"]["pct_by_category"]
    ax[0, 1].barh(list(cd.keys()), list(cd.values()), color="C2")
    for i, v in enumerate(cd.values()):
        ax[0, 1].text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax[0, 1].set_xlabel("cProfile 内部时间 %")
    ax[0, 1].set_title("跟踪器负载成本分解")
    ax[0, 1].grid(alpha=0.3, axis="x")

    # 3 共享扩展
    ss = report["sharing_scaling"]
    Ns = sorted(int(k) for k in ss)

    def ssval(n, key):
        return ss[n][key] if n in ss else ss[str(n)][key]

    ax[0, 2].plot(Ns, [ssval(n, "independent_us_per_block") for n in Ns],
                  "o-", label="独立检测")
    ax[0, 2].plot(Ns, [ssval(n, "shared_us_per_block") for n in Ns],
                  "s-", label="重叠共享")
    ax[0, 2].set_xlabel("重叠节点数")
    ax[0, 2].set_ylabel("µs / 块（N 节点合计）")
    ax[0, 2].set_title("频带重叠时共享检测计算")
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(alpha=0.3)

    # 4 交叉增益轨迹（B 节点，交点附近）
    m = (tt >= tc - 0.03) & (tt <= tc + 0.08)
    ax[1, 0].plot(tt[m] * 1000, r_live["gain_db"][1][m], "k-", lw=1.6,
                  label="逐样本更新参照")
    ax[1, 0].plot(tt[m] * 1000, r_ver["gain_db"][1][m], "o", ms=2.5,
                  label="轨迹版本缓存（重合）")
    ax[1, 0].plot(tt[m] * 1000, r_idx["gain_db"][1][m], "s-", ms=3,
                  color="C3", label="按频带编号缓存")
    ax[1, 0].axvline(tc * 1000, color="gray", ls=":", label="块中央交点")
    seg = report["crossing_probe"]["nodes"]["B"][
        "index_vs_version_first_bad_0p5db"]
    if seg:
        ax[1, 0].axvspan(seg["t0"] * 1000, seg["t1"] * 1000, color="red",
                         alpha=0.15, label="交点后首段错误衰减")
    ax[1, 0].set_xlabel("时间 ms")
    ax[1, 0].set_ylabel("B 节点增益 dB")
    ax[1, 0].set_title("轨迹交叉：编号键 vs 轨迹版本键")
    ax[1, 0].legend(fontsize=7)
    ax[1, 0].grid(alpha=0.3)

    # 5 短事件命中率
    se = report["short_events"]
    polls = [1, 4, 8, 16]
    hits = [se[f"poll{p}_hits"] for p in polls]
    ax[1, 1].plot([f"逐块", "4块一查", "8块一查", "16块一查"], hits, "o-")
    ax[1, 1].axhline(se["n_bursts"], color="g", ls=":", label="真实短峰数")
    ax[1, 1].axhline(se["kernel_env_hits"], color="C3", ls="--",
                     label="内核逐样本包络捕获")
    ax[1, 1].set_ylim(0, se["n_bursts"] + 2)
    ax[1, 1].set_ylabel("命中数")
    ax[1, 1].set_title("不能靠降低检测频率提速（2ms 短峰）")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    # 6 频段响应实测 vs 模型
    freqs = mp["freqs"]
    w0 = report["band_response"]["windows"][0]
    ax[1, 2].semilogx(freqs, w0["model_db"], "-", color="C0", label="RBJ 模型")
    ax[1, 2].semilogx(freqs, w0["meas_db"], "o", color="C3", label="实测投影")
    ax[1, 2].axhline(0, color="gray", ls=":")
    ax[1, 2].set_xlabel("Hz")
    ax[1, 2].set_ylabel("增益 dB")
    ax[1, 2].set_title(f"频段响应（最大模型误差 "
                       f"{w0['max_model_err_db']:.2f} dB）")
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "P8_detect_speedup.png"), dpi=130)
    plt.close(fig)

    # P9：交叉全轨迹 + 湿声差
    fig2, ax2 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    mm = (tt >= 0.15) & (tt <= 0.34)
    ax2[0].plot(tt[mm], r_ver["fdet"][0][mm], label="A 轨迹")
    ax2[0].plot(tt[mm], r_ver["fdet"][1][mm], label="B 轨迹")
    ax2[0].axvline(tc, color="k", ls=":")
    ax2[0].set_ylabel("检测频率 Hz")
    ax2[0].set_title("两条同起点、不同自动化曲线在块中央相交")
    ax2[0].legend(fontsize=8)
    ax2[0].grid(alpha=0.3)
    ax2[1].plot(tt[mm], (r_idx["y"] - r_live["y"])[mm], color="C3", lw=0.8,
                label="编号缓存湿声误差（相对逐样本）")
    ax2[1].plot(tt[mm], (r_ver["y"] - r_live["y"])[mm], color="C0", lw=0.8,
                label="版本缓存湿声误差（恒为 0）")
    ax2[1].axvline(tc, color="k", ls=":")
    ax2[1].set_ylabel("湿声差")
    ax2[1].set_xlabel("时间 s")
    ax2[1].legend(fontsize=8)
    ax2[1].grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGS, "P9_crossing_cache.png"), dpi=130)
    plt.close(fig2)


if __name__ == "__main__":
    main()
