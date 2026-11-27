"""检测计算提速的实验与复用原语：轨迹交叉探针、缓存键策略、共享检测池。

本模块只做三件可核验的事：
1. 提供逐样本系数更新的参照引擎，以及两种缓存键策略：
   - "version"：键含轨迹版本与完整系数签名（f/Q/g/fs），轨迹交叉不会错配，
     轨迹相同仅显示名不同的节点必然命中同一条目；
   - "index"：按频带（排序）编号取系数的反例缓存，交叉后编号易主时会把
     另一条轨迹的系数发给当前节点，用于定位"交点之后第一段错误衰减"。
2. SharedDetectionBank：相同检测频段在同一块输入上共享同一次 lfilter；
   自动化频移产生新键 -> 部分共享失效，split 时迁移滤波器状态，与独立
   计算逐位一致。
3. first_bad_segment：在误差曲线上找指定时刻之后第一段超差区间。

标量 biquad 与 RBJ 系数公式和 dyneq.kernel 完全一致，保证实验结论可回指内核。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.signal import lfilter

from .kernel import bandpass_coeffs, peaking_coeffs

LAB_BLOCK = 32


# ---------------------------------------------------------------- 轨迹
def hold_then_ramp(keyframes):
    """分段线性自动化。keyframes: [(t_s, f_hz), ...]，段外保持端点值。"""
    kf = sorted(keyframes)
    ts = np.array([p[0] for p in kf])
    fs_ = np.array([p[1] for p in kf])

    def f(t):
        return float(np.interp(t, ts, fs_))

    return f


def make_crossing_trajectories(fs=48000, block=128, f0=900.0, fc=2200.0,
                               d=500.0, b_cross=90):
    """两条同起点、不同曲线且在某个块正中央精确相交的自动化。

    交叉局部采用对称斜率：在交叉块块首两频率为 fc±d（相距 2d），
    块中央同时到达 fc，块末互换到 fc∓d。这样"按频带排序编号"的缓存
    在交叉后会把块首为另一条轨迹计算的系数发给当前节点，错配量可达 d。
    全局曲线仍从同一 f0 起步：B 长保持 f0 后陡升；A 先快升到高台再下降
    穿过 fc。返回的 f_sig=fc+d 是放大错配可听效果的建议声源频率。
    """
    tc_sample = b_cross * block + block // 2
    tc = tc_sample / fs
    half = block // 2 / fs
    v = d / half                          # Hz/s
    f_hi = fc + 2.0 * d                   # 3200
    f_lo = fc - 2.0 * d                   # 1200
    # B：+v 直线穿过 (tc,fc)，反推到 f0 的时刻，此前保持 f0
    tB0 = tc - (fc - f0) / v
    tB0 = max(tB0, 0.06)

    def fB(t):
        return f0 if t < tB0 else fc + v * (t - tc)

    # A：先 f0 保持、快升到 f_hi 平台，再以 -v 下降穿过 (tc,fc)
    t_top = tc - (f_hi - fc) / v
    t_bot = tc + (fc - f_lo) / v
    fA = hold_then_ramp([
        (0.0, f0), (0.05, f0), (0.10, f_hi), (t_top, f_hi),
        (tc, fc), (t_bot, f_lo), (tc + 1.0, f_lo)])
    f_sig = fc + d
    info = {"t_cross": tc, "cross_sample": tc_sample, "cross_block": b_cross,
            "block": block, "f_cross": fc, "f0": f0, "d_hz": d,
            "f_at_block_start": [fc + d, fc - d],
            "f_at_block_end": [fc - d, fc + d],
            "slope_mag_hz_s": v, "f_sig": f_sig,
            "order_before": "A>B", "t_B_ramp_start": tB0,
            "note": "对称陡交叉：块首 fc±d，块中央相交，块末互换"}
    return fA, fB, info


# ---------------------------------------------------------------- 标量 biquad
class Biquad:
    __slots__ = ("b0", "b1", "b2", "a1", "a2", "x1", "x2", "y1", "y2")

    def __init__(self):
        self.b0 = 1.0
        self.b1 = self.b2 = self.a1 = self.a2 = 0.0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def set(self, c):
        b, a = c[0], c[1]
        self.b0, self.b1, self.b2 = float(b[0]), float(b[1]), float(b[2])
        self.a1, self.a2 = float(a[1]), float(a[2])

    def run(self, x):
        y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
             - self.a1 * self.y1 - self.a2 * self.y2)
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y


@dataclass
class LabNode:
    name: str
    trajectory: Callable[[float], float]
    band_no: int = 0
    traj_version: str = "curve-v1"
    q_det: float = 6.0
    q_proc: float = 3.0
    threshold_db: float = -30.0
    ratio: float = 4.0
    max_cut_db: float = 16.0
    det_attack_ms: float = 0.2
    det_release_ms: float = 40.0
    attack_ms: float = 0.2
    release_ms: float = 20.0
    proc_trajectory: Optional[Callable[[float], float]] = None
    display_name_only: str = ""     # 不进入任何缓存键，专供"仅换显示名"对照


class VersionCoeffCache:
    """键 = (轨迹版本, 种类, 完整系数签名)。显示名不在键中。"""

    def __init__(self):
        self.store = {}
        self.hits = 0
        self.misses = 0

    def bandpass(self, node, f, fs):
        key = ("bp", node.traj_version, float(f), float(node.q_det), float(fs))
        c = self.store.get(key)
        if c is None:
            c = bandpass_coeffs(f, node.q_det, fs)
            self.store[key] = c
            self.misses += 1
        else:
            self.hits += 1
        return c

    def stats(self):
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else 1.0, "entries": len(self.store)}


class IndexSwapCache:
    """反例：块内只按"当前频率排序得到的频带编号"缓存检测系数。

    每个块开始时清空 held；处理每个样本时按各节点当前 f 排序确定编号，
    编号在该样本第一次被请求时按"当前节点的当前 f"计算系数，之后整块复用。
    轨迹在块中央交叉 -> 排序易主 -> 交叉后新编号持有者拿到交叉前另一条
    轨迹的系数。这正是"按频带编号缓存、不含轨迹版本"的典型错配。
    """

    def __init__(self, n_nodes):
        self.n = n_nodes
        self.held = {}
        self.held_f = {}
        self.requests = 0
        self.max_freq_mismatch = 0.0   # 命中条目系数频率与当前请求 f 的最大偏差

    def begin_block(self):
        self.held.clear()
        self.held_f.clear()

    @staticmethod
    def rank_of(f_now, j):
        order = sorted(range(len(f_now)), key=lambda k: (f_now[k], k))
        return order.index(j)

    def bandpass(self, node, j, f_now, fs):
        no = self.rank_of(f_now, j)
        key = ("bp", no)
        self.requests += 1
        c = self.held.get(key)
        if c is None:
            c = bandpass_coeffs(f_now[j], node.q_det, fs)
            self.held[key] = c
            self.held_f[key] = float(f_now[j])
        else:
            self.max_freq_mismatch = max(
                self.max_freq_mismatch, abs(self.held_f[key] - f_now[j]))
        return c


def _env_gain_constants(node, fs):
    kae = 1.0 - math.exp(-1.0 / (node.det_attack_ms * 1e-3 * fs))
    kre = 1.0 - math.exp(-1.0 / (node.det_release_ms * 1e-3 * fs))
    kag = 1.0 - math.exp(-1.0 / (node.attack_ms * 1e-3 * fs))
    krg = 1.0 - math.exp(-1.0 / (node.release_ms * 1e-3 * fs))
    return kae, kre, kag, krg


def run_nodes(nodes, x, fs, mode="live", version_cache=None, block=LAB_BLOCK):
    """串行多节点动态均衡标量引擎。

    mode:
      - "live":    每个样本重算检测系数（逐样本更新参照）
      - "version": 经 VersionCoeffCache 取系数（与 live 逐位一致，可跨节点共享）
      - "hold":    稳定节点键：每块首样本算一次并整块保持（正确的块率近似）
      - "index":   经 IndexSwapCache 取检测系数（按频带编号的反例）
    返回 dict: y, gain[N,n], env[N,n], fdet[N,n], cache(对象或统计)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    N = len(nodes)
    det_bq = [Biquad() for _ in range(N)]
    proc_bq = [Biquad() for _ in range(N)]
    env = [0.0] * N
    gain = [0.0] * N
    held_bp = [None] * N
    cs = [_env_gain_constants(nd, fs) for nd in nodes]
    ratio_gain = [1.0 - 1.0 / max(nd.ratio, 1.0) for nd in nodes]
    gains = np.zeros((N, n))
    envs = np.zeros((N, n))
    fdet_log = np.zeros((N, n))
    idx_cache = IndexSwapCache(N) if mode == "index" else None
    vc = version_cache if (mode == "version" and version_cache is not None) \
        else (VersionCoeffCache() if mode == "version" else None)

    y = x.astype(np.float64).copy()
    for i in range(n):
        t = i / fs
        f_now = [max(nd.trajectory(t), 1.0) for nd in nodes]
        if i % block == 0:
            if mode == "index":
                idx_cache.begin_block()
            if mode == "hold":
                for j, nd in enumerate(nodes):
                    held_bp[j] = bandpass_coeffs(f_now[j], nd.q_det, fs)
        # ---- 检测 + 增益判决：所有检测节点并行拾取同一原始声源
        for j, nd in enumerate(nodes):
            fdet = f_now[j]
            fdet_log[j, i] = fdet
            if mode == "live":
                c_bp = bandpass_coeffs(fdet, nd.q_det, fs)
            elif mode == "version":
                c_bp = vc.bandpass(nd, fdet, fs)
            elif mode == "hold":
                c_bp = held_bp[j]
            else:
                c_bp = idx_cache.bandpass(nd, j, f_now, fs)
            det_bq[j].set(c_bp)
            d = det_bq[j].run(x[i])
            v = abs(d)
            kae, kre, kag, krg = cs[j]
            env[j] += (kae if v > env[j] else kre) * (v - env[j])
            over = 20.0 * math.log10(env[j] + 1e-9) - nd.threshold_db
            cut = over * ratio_gain[j] if over > 0.0 else 0.0
            if cut > nd.max_cut_db:
                cut = nd.max_cut_db
            tgt = -cut
            gain[j] += (kag if tgt < gain[j] else krg) * (tgt - gain[j])
            gains[j, i] = gain[j]
            envs[j, i] = env[j]
            fp = nd.proc_trajectory(t) if nd.proc_trajectory else fdet
            proc_bq[j].set(peaking_coeffs(gain[j], fp, nd.q_proc, fs))
        # ---- 串行处理：节点 j 的峰形滤波作用于节点 j-1 的当前样本输出
        ys = x[i]
        for j in range(N):
            ys = proc_bq[j].run(ys)
        y[i] = ys
    return {"y": y, "gain_db": gains, "env": envs, "fdet": fdet_log,
            "version_cache": vc.stats() if vc is not None else None,
            "index_cache": {"requests": idx_cache.requests,
                            "max_freq_mismatch_hz": idx_cache.max_freq_mismatch}
                           if idx_cache is not None else None}


def first_bad_segment(t, err_db, t_after, tol_db=0.5):
    """t>=t_after 之后第一段连续 err_db>tol_db 的区间。找不到返回 None。"""
    t = np.asarray(t)
    err = np.asarray(err_db)
    cand = np.where((t >= t_after) & (err > tol_db))[0]
    if cand.size == 0:
        return None
    i0 = int(cand[0])
    i1 = i0
    while i1 + 1 < err.size and err[i1 + 1] > tol_db:
        i1 += 1
    seg = err[i0:i1 + 1]
    return {"i0": i0, "i1": i1, "t0": float(t[i0]), "t1": float(t[i1]),
            "n_samples": i1 - i0 + 1,
            "peak_err_db": float(seg.max()), "mean_err_db": float(seg.mean())}


# ---------------------------------------------------------------- 共享检测池
class SharedDetectionBank:
    """相同键 (f,q,fs,version) 的检测带通在同一块上只执行一次 lfilter。

    条目保存自己的 zi。频移后请求新键 -> miss；未移动的节点继续命中旧键，
    即"部分共享失效"。split_state 可在失效边界把旧键 zi 迁移给新键，
    使新通路与一直独立计算的滤波器状态逐位相同。
    """

    def __init__(self, fs):
        self.fs = float(fs)
        self.entries = {}
        self.lfilter_runs = 0
        self.block_requests = 0
        self.hits = 0

    def _ensure(self, key, f, q):
        e = self.entries.get(key)
        if e is None:
            b, a = bandpass_coeffs(f, q, self.fs)
            e = {"b": b, "a": a, "zi": np.zeros(max(len(a), len(b)) - 1),
                 "f": float(f), "q": float(q)}
            self.entries[key] = e
        return e

    def split_state(self, old_key, new_key, f_new, q_new):
        """频移失效边界：把旧键的滤波器状态迁移到新键（复制，不破坏旧键）。"""
        old = self.entries[old_key]
        e = self._ensure(new_key, f_new, q_new)
        e["zi"] = old["zi"].copy()

    def process_block(self, key, f, q, blk):
        self.block_requests += 1
        e = self._ensure(key, f, q)
        already = e.get("ran_block", False)
        # 同一键在同一块内被多个节点请求时，第二次起直接复用缓存输出
        if already:
            self.hits += 1
            return e["last_y"]
        y, zi = lfilter(e["b"], e["a"], blk, zi=e["zi"])
        e["zi"] = zi
        e["last_y"] = y
        e["ran_block"] = True
        self.lfilter_runs += 1
        return y

    def begin_block(self):
        for e in self.entries.values():
            e["ran_block"] = False


def independent_detector_block(states, tag, f, q, fs, blk):
    """对照用：每个节点独立 zi 的同一块 lfilter；f/q 变化时更新系数并保留 zi。"""
    e = states.get(tag)
    if e is None:
        b, a = bandpass_coeffs(f, q, fs)
        e = {"b": b, "a": a, "zi": np.zeros(max(len(a), len(b)) - 1),
             "f": float(f), "q": float(q)}
        states[tag] = e
    elif e["f"] != float(f) or e["q"] != float(q):
        e["b"], e["a"] = bandpass_coeffs(f, q, fs)
        e["f"], e["q"] = float(f), float(q)
    y, zi = lfilter(e["b"], e["a"], blk, zi=e["zi"])
    e["zi"] = zi
    return y


# ---------------------------------------------------------------- 包络/增益
def follow_detection(det, fs, det_attack_ms=0.2, det_release_ms=40.0):
    """与内核同式的逐样本峰值包络。"""
    ka = 1.0 - math.exp(-1.0 / (det_attack_ms * 1e-3 * fs))
    kr = 1.0 - math.exp(-1.0 / (det_release_ms * 1e-3 * fs))
    env = np.zeros_like(det)
    e = 0.0
    for i, v0 in enumerate(np.abs(det)):
        e += (ka if v0 > e else kr) * (v0 - e)
        env[i] = e
    return env


def gain_curve_from_env(env, fs, threshold_db=-30.0, ratio=4.0, max_cut_db=16.0,
                        attack_ms=0.2, release_ms=20.0):
    """与内核同式的逐样本增益平滑，返回 gain_db（<=0）。"""
    ka = 1.0 - math.exp(-1.0 / (attack_ms * 1e-3 * fs))
    kr = 1.0 - math.exp(-1.0 / (release_ms * 1e-3 * fs))
    rg = 1.0 - 1.0 / max(ratio, 1.0)
    g = 0.0
    out = np.zeros_like(env)
    for i, e0 in enumerate(env):
        over = 20.0 * math.log10(e0 + 1e-9) - threshold_db
        cut = over * rg if over > 0 else 0.0
        cut = min(cut, max_cut_db)
        tgt = -cut
        g += (ka if tgt < g else kr) * (tgt - g)
        out[i] = g
    return out


def peaking_wet_serial(x, gains, freqs, q_proc, fs):
    """按逐样本增益/频率跑串行峰形 biquad，供全量衰减测量。"""
    n = x.size
    N = gains.shape[0]
    bqs = [Biquad() for _ in range(N)]
    y = np.zeros(n)
    for i in range(n):
        ys = x[i]
        for j in range(N):
            f = float(freqs[j, i]) if freqs.ndim == 2 else float(freqs[j])
            bqs[j].set(peaking_coeffs(float(gains[j, i]), f, q_proc, fs))
            ys = bqs[j].run(ys)
        y[i] = ys
    return y


# ---------------------------------------------------------------- 重叠共享场景
def overlap_sharing_scenario(fs=48000, dur=1.0, shift_t=0.5, block=LAB_BLOCK,
                             f_common=3000.0, f_shifted=1800.0, q=8.0, seed=5):
    """两检测节点前半段同频（共享一次 lfilter），shift 块边界 B 频移。

    共享池路径与"每节点独立 zi + 频移时保留 zi"的参照路径逐块对照：
    det、包络、增益都应逐位相同。返回全量数组与共享计数，供测试与基准脚本
    做全量衰减核验。
    """
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    t = np.arange(n) / fs
    x = (0.28 * np.sin(2 * np.pi * f_common * t)
         + 0.22 * np.sin(2 * np.pi * f_shifted * t)
         + 0.01 * rng.standard_normal(n))
    shift_block = int(round(shift_t * fs / block))
    shift_sample = shift_block * block

    bank = SharedDetectionBank(fs)
    indep = {}
    det_sh = np.zeros((2, n))
    det_in = np.zeros((2, n))
    key0 = (f_common, q, fs, "traj-v1")
    key1 = (f_shifted, q, fs, "traj-v2-shift")
    migrated = False

    for b0 in range(0, n, block):
        blk = x[b0:min(b0 + block, n)]
        bi = b0 // block
        bank.begin_block()
        shifted = bi >= shift_block
        if shifted and not migrated:
            # 必须在本块任何 lfilter 之前迁移：zi 是上一块结束时的边界状态
            bank.split_state(key0, key1, f_shifted, q)
            migrated = True
        # 共享池
        yA = bank.process_block(key0, f_common, q, blk)
        if not shifted:
            yB = bank.process_block(key0, f_common, q, blk)   # 同键 -> 命中共享
        else:
            yB = bank.process_block(key1, f_shifted, q, blk)
        det_sh[0, b0:b0 + blk.size] = yA
        det_sh[1, b0:b0 + blk.size] = yB
        # 独立参照
        zA = independent_detector_block(indep, "A", f_common, q, fs, blk)
        fB = f_shifted if shifted else f_common
        zB = independent_detector_block(indep, "B", fB, q, fs, blk)
        det_in[0, b0:b0 + blk.size] = zA
        det_in[1, b0:b0 + blk.size] = zB

    env_sh = np.vstack([follow_detection(det_sh[j], fs) for j in range(2)])
    env_in = np.vstack([follow_detection(det_in[j], fs) for j in range(2)])
    g_sh = np.vstack([gain_curve_from_env(env_sh[j], fs) for j in range(2)])
    g_in = np.vstack([gain_curve_from_env(env_in[j], fs) for j in range(2)])
    freqs = np.zeros((2, n))
    freqs[0] = f_common
    freqs[1, :shift_sample] = f_common
    freqs[1, shift_sample:] = f_shifted
    wet_sh = peaking_wet_serial(x, g_sh, freqs, 3.0, fs)
    wet_in = peaking_wet_serial(x, g_in, freqs, 3.0, fs)
    return {"x": x, "wet_shared": wet_sh, "wet_indep": wet_in,
            "det_shared": det_sh, "det_indep": det_in,
            "gain_shared": g_sh, "gain_indep": g_in, "freqs": freqs,
            "shift_sample": shift_sample, "shift_t": shift_sample / fs,
            "bank_stats": {"lfilter_runs": bank.lfilter_runs,
                           "block_requests": bank.block_requests,
                           "cache_hits": bank.hits,
                           "unique_entries": len(bank.entries)},
            "max_det_diff": float(np.max(np.abs(det_sh - det_in))),
            "max_gain_diff": float(np.max(np.abs(g_sh - g_in))),
            "wet_bitexact": bool(np.array_equal(wet_sh, wet_in))}


# ---------------------------------------------------------------- 短事件探针
def short_burst_train(fs=48000, dur=0.5, f_tone=3000.0, burst_ms=2.0,
                      period_ms=17.0, amp=0.5):
    """一串窄于常见降采样检测间隔的短峰（2ms），周期与轮询栅格错开。"""
    n = int(dur * fs)
    t = np.arange(n) / fs
    x = np.zeros(n)
    L = int(burst_ms * 1e-3 * fs)
    P = int(period_ms * 1e-3 * fs)
    starts = list(range(int(0.03 * fs), n - L, P))
    env = np.hanning(2 * L)[:L] if L > 2 else np.ones(L)
    for s0 in starts:
        tt = np.arange(L) / fs
        x[s0:s0 + L] += amp * env * np.sin(2 * np.pi * f_tone * tt)
    return x, np.array(starts), L


def band_block_levels(x, fs, f_hz=3000.0, q=10.0, block=LAB_BLOCK, poll_blocks=1):
    """带通检测后逐块（或每 poll_blocks 块才采样一次）的峰值电平。

    poll_blocks>1 模拟"为提速降低检测频率"：只在轮询块更新读数，中间事件
    可能完全落在两次轮询之间而被漏掉。返回 (levels, block_times)。
    """
    b, a = bandpass_coeffs(f_hz, q, fs)
    zi = np.zeros(max(len(a), len(b)) - 1)
    nb = (x.size + block - 1) // block
    levels, times = [], []
    for bi in range(nb):
        i0, i1 = bi * block, min((bi + 1) * block, x.size)
        y, zi = lfilter(b, a, x[i0:i1], zi=zi)
        if bi % poll_blocks == 0 or bi == nb - 1:
            levels.append(float(np.max(np.abs(y))))
            times.append(i0 / fs)
    return np.array(levels), np.array(times)


def count_detected_bursts(levels, times, starts, fs, threshold=0.05):
    """检测电平超阈值且时间上对应某个真实短峰的命中数（去重）。"""
    hot = times[levels > threshold]
    hit = 0
    for s0 in starts:
        tc = s0 / fs
        if hot.size and np.min(np.abs(hot - tc)) < 0.01:
            hit += 1
    return hit

