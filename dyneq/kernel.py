"""动态均衡内核：窄带检测驱动单个参量滤波器的增益。

设计约束：
- 严格因果、单遍处理：已输出的样本永不重算（录音安全）。
- 只改变对应参量峰形滤波器的增益，不做整段频带以外的宽带压低。
- 检测中心与处理中心可分别设置；处理中心也可选择跟随检测中心加偏移。
- 所有参数经内核校验/钳位，自动化路径与界面路径走同一入口，非法值无法绕过。
- 系数逐控制块更新，参数先平滑再算系数，并逐块检查极点半径。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter, firwin

F_MIN = 20.0          # 中心频率下限：低于此 sin(w0)->0，alpha->0，极点半径->1
NYQ_GUARD = 0.45      # 中心频率上限 = 0.45*fs：靠近奈奎斯特时频率翘曲且 sin(w0)->0
R_MAX = 0.99995       # 极点半径安全上限
Q_MIN, Q_MAX = 0.2, 24.0
G_MIN, G_MAX = -30.0, 0.0   # 动态段只允许衰减（dB）
EPS = 1e-12


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def pole_radius(a1, a2):
    """二阶分母 1 + a1 z^-1 + a2 z^-2 的极点半径。"""
    disc = a1 * a1 - 4.0 * a2
    if disc < 0:
        return math.sqrt(max(a2, 0.0))
    s = math.sqrt(disc)
    return max(abs((-a1 + s) / 2), abs((-a1 - s) / 2))


def peaking_coeffs(g_db, f_hz, q, fs):
    """RBJ 峰形滤波器。返回 (b, a, 极点半径, 是否触发安全钳位)。"""
    g_db = _clamp(g_db, G_MIN, G_MAX)
    f_hz = _clamp(f_hz, F_MIN, NYQ_GUARD * fs)
    q = _clamp(q, Q_MIN, Q_MAX)
    A = 10.0 ** (g_db / 40.0)
    w0 = 2.0 * math.pi * f_hz / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    a0 = 1.0 + alpha / A
    b = np.array([(1.0 + alpha * A) / a0, -2.0 * cw / a0, (1.0 - alpha * A) / a0])
    a = np.array([1.0, -2.0 * cw / a0, (1.0 - alpha / A) / a0])
    r = pole_radius(a[1], a[2])
    clamped = False
    if r > R_MAX:  # 理论上钳位参数后不会触发，兜底保证连续自动化下也不越界
        k = R_MAX / r
        a[1] *= k
        a[2] *= k * k
        r = R_MAX
        clamped = True
    return b, a, r, clamped


def bandpass_coeffs(f_hz, q, fs):
    """RBJ 带通（峰值 0 dB），用于检测通路。"""
    f_hz = _clamp(f_hz, F_MIN, NYQ_GUARD * fs)
    q = _clamp(q, 0.3, 12.0)
    w0 = 2.0 * math.pi * f_hz / fs
    alpha = math.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    b = np.array([alpha / a0, 0.0, -alpha / a0])
    a = np.array([1.0, -2.0 * math.cos(w0) / a0, (1.0 - alpha) / a0])
    return b, a


class CoeffCache:
    """按"完整系数签名"取系数的备忘录：参数不变就不重算。

    键必须包含所有进入系数的量（种类、fs、f、Q、增益）。自动化改变频率后
    签名随之改变 -> 缓存未命中并按实际参数重算；仅更换显示名称等不进入签名
    的标签 -> 必然命中，正确复用。

    version 可选地带上"轨迹版本"：当同编号频带的自动化曲线被整体替换时，
    即使频率恰好相同也能用 version 防止旧曲线系数被复用（对照
    scripts/bench_detect_speed.py 中的"按频带编号"错误缓存）。
    本缓存只做备忘录：命中返回的系数与直接调用工厂函数逐位相同。
    """

    def __init__(self):
        # kind -> [签名, 结果, 命中数, 未命中数]；每类仅一个槽，严格有界
        self._slot = {}
        self.hits = 0
        self.misses = 0

    def get(self, kind, sig, factory, *fargs, version=None):
        key = sig if version is None else (version, sig)
        slot = self._slot.get(kind)
        if slot is not None and slot[0] == key:
            slot[2] += 1
            self.hits += 1
            return slot[1]
        obj = factory(*fargs)
        if slot is None:
            self._slot[kind] = [key, obj, 0, 1]
        else:
            slot[0] = key
            slot[1] = obj
            slot[3] += 1
        self.misses += 1
        return obj

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "lookups": total,
                "hit_rate": (self.hits / total) if total else 1.0,
                "kind_hits": {k: s[2] for k, s in self._slot.items()},
                "kind_misses": {k: s[3] for k, s in self._slot.items()},
                "slots": len(self._slot)}

    def reset_stats(self):
        self.hits = self.misses = 0
        for slot in self._slot.values():
            slot[2] = 0
            slot[3] = 0


@dataclass
class ParamSpec:
    name: str
    kind: str          # 'float' | 'bool'
    lo: float
    hi: float
    default: float
    unit: str
    note: str


class DynamicEQ:
    """单频段动态均衡内核（块处理，控制率 = fs/BLOCK）。"""

    BLOCK = 32
    N_TRACK_BANDS = 13

    def __init__(self, fs=48000):
        self.fs = float(fs)
        fmax = NYQ_GUARD * self.fs
        self._schema = [
            ParamSpec("detect_freq", "float", F_MIN, fmax, 3000.0, "Hz",
                      "检测中心频率。低于20Hz或高于0.45fs时 sin(w0)->0、极点半径->1、频率翘曲，内核强制钳位"),
            ParamSpec("detect_q", "float", 0.5, 8.0, 2.0, "", "检测带通 Q"),
            ParamSpec("process_freq", "float", F_MIN, fmax, 3000.0, "Hz",
                      "处理中心频率（process_follow 关闭时生效），与检测中心相互独立"),
            ParamSpec("process_q", "float", Q_MIN, Q_MAX, 3.0, "", "处理峰形 Q；Q 过大使极点半径->1"),
            ParamSpec("process_follow", "bool", 0, 1, 0, "", "处理中心是否跟随检测中心"),
            ParamSpec("process_offset_hz", "float", -4000, 4000, 0.0, "Hz", "跟随模式下处理中心相对检测中心的偏移"),
            ParamSpec("threshold_db", "float", -80, 0, -30.0, "dBFS", "检测包络阈值"),
            ParamSpec("ratio", "float", 1.0, 10.0, 3.0, ":1", "超阈值衰减比"),
            ParamSpec("max_cut_db", "float", 0.0, 30.0, 18.0, "dB", "最大衰减量"),
            ParamSpec("attack_ms", "float", 0.1, 500, 10.0, "ms", "增益包络启动时间"),
            ParamSpec("release_ms", "float", 1.0, 2000, 180.0, "ms", "增益包络释放时间"),
            ParamSpec("det_attack_ms", "float", 0.1, 100, 4.0, "ms", "检测包络启动时间"),
            ParamSpec("det_release_ms", "float", 1.0, 1000, 90.0, "ms", "检测包络释放时间"),
            ParamSpec("mix", "float", 0.0, 1.0, 1.0, "", "湿声比例（1=全处理，不做响度归一化）"),
            ParamSpec("tracker_enable", "bool", 0, 1, 0, "", "相邻峰跟踪器开关"),
            ParamSpec("track_range_hz", "float", 100, 4000, 900.0, "Hz", "跟踪搜索范围（检测中心±）"),
            ParamSpec("dwell_ms", "float", 0.0, 500, 60.0, "ms",
                      "驻留时间：检测窗尚未看全新峰时保持旧中心、暂缓跟随"),
            ParamSpec("switch_margin_db", "float", 0.0, 12.0, 3.0, "dB", "挑战峰进入候选的余量"),
            ParamSpec("lost_hold_ms", "float", 0.0, 2000, 400.0, "ms", "原声部消失多久后重新捕获"),
            ParamSpec("f_slew_ms", "float", 1.0, 200, 25.0, "ms", "中心频率切换的斜变时间"),
        ]
        self.params = {s.name: float(s.default) for s in self._schema}
        self.feedback = []          # 内核 -> 界面 的反馈（钳位/拒绝/事件）
        self.events = []            # 跟踪器事件
        self.pole_clamp_count = 0   # 极点安全钳位触发次数
        self.coeff_cache = CoeffCache()  # 检测带通/处理峰形系数缓存（按完整签名失效）
        self._reset_state()

    def _cached_bandpass(self, f_hz, q):
        return self.coeff_cache.get(
            "bandpass", (float(f_hz), float(q), self.fs),
            bandpass_coeffs, f_hz, q, self.fs)

    def _cached_peaking(self, g_db, f_hz, q):
        return self.coeff_cache.get(
            "peaking", (float(g_db), float(f_hz), float(q), self.fs),
            peaking_coeffs, g_db, f_hz, q, self.fs)

    # ------------------------------------------------------------------ 参数
    def schema(self):
        return list(self._schema)

    def set_params(self, **kw):
        """唯一参数入口（界面与自动化共用）。返回 (name, status, message) 列表。

        status: ok / clamped / rejected。非法值（NaN、inf、类型错误）拒绝；
        越界值钳位并给出物理解释，绝不被内部自动化静默提交。
        """
        out = []
        for name, val in kw.items():
            spec = next((s for s in self._schema if s.name == name), None)
            if spec is None:
                out.append((name, "rejected", f"未知参数 {name}"))
                continue
            if spec.kind == "bool":
                self.params[name] = 1.0 if bool(val) else 0.0
                out.append((name, "ok", ""))
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                out.append((name, "rejected", f"非数值 {val!r}"))
                continue
            if not math.isfinite(v):
                out.append((name, "rejected", f"非有限值 {val!r}，已拒绝"))
                continue
            if v < spec.lo or v > spec.hi:
                c = _clamp(v, spec.lo, spec.hi)
                msg = self._clamp_note(spec, v, c)
                self.params[name] = c
                out.append((name, "clamped", msg))
                self.feedback.append(f"[clamp] {msg}")
            else:
                self.params[name] = v
                out.append((name, "ok", ""))
        if any(n in kw for n in ("detect_freq", "track_range_hz", "tracker_enable")) \
                and self.params["tracker_enable"] > 0.5:
            self._tracker_reset()
        if self._sample_idx == 0:
            # 处理前的配置直接对齐平滑状态；处理中的自动化走斜变（防爆音）
            self._f_det = self.params["detect_freq"]
            self._f_proc = self._proc_target()
        return out

    @staticmethod
    def _clamp_note(spec, v, c):
        if spec.name.endswith("freq") or spec.name == "process_offset_hz":
            return (f"{spec.name}={v:g} 越界，已钳位到 {c:g} {spec.unit}："
                    f"中心频率趋近 0 时 sin(w0)->0 使 alpha->0、极点半径->1（建立时间发散、峰形退化）；"
                    f"趋近奈奎斯特时频率翘曲使实际中心偏离且包络更新失稳，故内核限制在 "
                    f"[{spec.lo:g}, {spec.hi:g}]")
        if spec.name.endswith("_q"):
            return (f"{spec.name}={v:g} 越界，已钳位到 {c:g}："
                    f"Q 过大使极点半径->1（振铃发散），过小则峰形失去意义")
        return f"{spec.name}={v:g} 越界，已钳位到 {c:g}（允许范围 [{spec.lo:g}, {spec.hi:g}]）"

    # ------------------------------------------------------------------ 状态
    def _reset_state(self):
        p = self.params
        self._zi_det = np.zeros(2)
        self._zi_proc = np.zeros(2)
        self._env = 0.0            # 检测包络
        self._gain_db = 0.0        # 平滑后的增益
        self._f_det = p["detect_freq"]
        self._f_proc = self._proc_target()
        self._sample_idx = 0
        self._tracker_reset(log=False)

    def _tracker_reset(self, log=True):
        p = self.params
        lo = _clamp(p["detect_freq"] - p["track_range_hz"], F_MIN, NYQ_GUARD * self.fs)
        hi = _clamp(p["detect_freq"] + p["track_range_hz"], F_MIN, NYQ_GUARD * self.fs)
        if hi <= lo * 1.05:
            hi = min(lo * 1.05, NYQ_GUARD * self.fs)
        self._trk_centers = np.geomspace(lo, hi, self.N_TRACK_BANDS)
        bw = (hi - lo) / (self.N_TRACK_BANDS - 1)
        r = math.exp(-math.pi * max(bw, 1.0) / self.fs)
        self._trk_r = r
        # 13 个复数谐振器的系数只取决于中心频率与 r（仅 tracker_reset 时变化），
        # 在此预计算一次并复用，避免每个控制块重复进行 13 组三角/数组构造。
        self._trk_res_b = np.array([1.0 - r])
        self._trk_res_a = []
        for m in range(self.N_TRACK_BANDS):
            w = 2.0 * math.pi * self._trk_centers[m] / self.fs
            self._trk_res_a.append(
                np.array([1.0, -r * complex(math.cos(w), math.sin(w))]))
        self._trk_zi = [np.zeros(1, dtype=complex) for _ in range(self.N_TRACK_BANDS)]
        self._trk_env = np.zeros(self.N_TRACK_BANDS)
        # 包络平滑系数按块率计算（每块 BLOCK 个样本）
        self._trk_env_k = 1.0 - math.exp(-self.BLOCK / (0.008 * self.fs))
        self._trk_fc = p["detect_freq"]
        self._trk_fc_target = p["detect_freq"]
        self._trk_timer = 0
        self._trk_base = np.zeros(self.N_TRACK_BANDS)   # 各频带慢速基线（上升沿门控）
        self._trk_voice_ref = 0.0                        # 当前声部包络基线（retention 分母）
        self._trk_lost = 0
        self._trk_hold_until = 0.0
        self._trk_last_hold_log = -1.0
        # 相位证据链：带通 FIR + Hilbert FIR + 对齐延迟 -> 解析信号瞬时频率
        bp_lo = max(lo, 40.0)
        bp_hi = min(hi, self.fs * NYQ_GUARD)
        self._ev_bp = firwin(129, [bp_lo, bp_hi], pass_zero=False, fs=self.fs)
        n_h = np.arange(-31, 32)
        h = np.zeros(63)
        odd = n_h % 2 != 0
        h[odd] = 2.0 / (np.pi * n_h[odd])
        self._ev_hil = h * np.hamming(63)
        self._ev_delay = np.zeros(32)
        self._ev_delay[31] = 1.0
        self._ev_zi_bp = np.zeros(128)
        self._ev_zi_hil = np.zeros(62)
        self._ev_zi_dl = np.zeros(31)
        self._ev_fhist = np.zeros(0)
        self._trk_wobble = 0.0
        if log:
            self.events.append({"t": self._sample_idx / self.fs, "type": "tracker_reset",
                                "detail": f"band {lo:.0f}..{hi:.0f} Hz"})

    def _proc_target(self):
        p = self.params
        if p["process_follow"] > 0.5:
            fc = self._trk_fc if p["tracker_enable"] > 0.5 else p["detect_freq"]
            return _clamp(fc + p["process_offset_hz"], F_MIN, NYQ_GUARD * self.fs)
        return _clamp(p["process_freq"], F_MIN, NYQ_GUARD * self.fs)

    # ------------------------------------------------------------------ 跟踪
    def _tracker_step(self, x_blk, t_now):
        """复数谐振器组 + 驻留/证据判决。全部因果，无前瞻。

        证据：
        - retention：旧频带包络 / 挑战开始时的包络。能量迁移（滑移）时趋 0，
          新声部起振（旧音继续）时保持 ≈1。
        - wobble：解析信号瞬时频率抖动。单音（含滑移）相位连续 -> 低；
          双音并存产生差频摆动 -> 高。
        """
        p = self.params
        r = self._trk_r
        b = self._trk_res_b
        envs = np.empty(self.N_TRACK_BANDS)
        for m in range(self.N_TRACK_BANDS):
            y, self._trk_zi[m] = lfilter(b, self._trk_res_a[m], x_blk,
                                          zi=self._trk_zi[m])
            envs[m] = np.abs(y[-1]) if y.size else 0.0
        self._trk_env += self._trk_env_k * (envs - self._trk_env)

        # 相位证据：解析信号的瞬时频率标准差（约 2.7 ms 窗）
        xb, self._ev_zi_bp = lfilter(self._ev_bp, [1.0], x_blk, zi=self._ev_zi_bp)
        im, self._ev_zi_hil = lfilter(self._ev_hil, [1.0], xb, zi=self._ev_zi_hil)
        re, self._ev_zi_dl = lfilter(self._ev_delay, [1.0], xb, zi=self._ev_zi_dl)
        z = re + 1j * im
        if z.size > 1:
            f_inst = np.angle(z[1:] * np.conj(z[:-1])) * self.fs / (2.0 * math.pi)
            self._ev_fhist = np.concatenate([self._ev_fhist, f_inst])[-128:]
            if self._ev_fhist.size >= 64:
                wob = float(np.std(self._ev_fhist))
                self._trk_wobble += 0.2 * (wob - self._trk_wobble)

        cur = int(np.argmin(np.abs(self._trk_centers - self._trk_fc)))
        margin_lin = 10.0 ** (-p["switch_margin_db"] / 10.0)
        blk_ms = self.BLOCK / self.fs * 1000.0
        state = "idle"

        # 未解释能量：各频带包络减去"当前声部裙边可解释的部分"。
        # 稳态单音时相邻带能量都能被裙边解释 -> 不触发挑战；
        # 新声部出现时其频带能量超出裙边预期 -> 成为挑战者。
        dw = 2.0 * math.pi * (self._trk_centers - self._trk_fc) / self.fs
        skirt = (1.0 - r) / np.abs(1.0 - r * np.exp(1j * dw))
        expected = self._trk_env[cur] * skirt / max(skirt[cur], EPS)
        unexplained = self._trk_env - expected
        unexplained[cur] = -np.inf
        chal = int(np.argmax(unexplained))

        # 慢速基线：声部包络（retention 分母）与各频带包络（上升沿门控）
        self._trk_voice_ref = max(self._trk_env[cur],
                                  self._trk_voice_ref * math.exp(-blk_ms / 3000.0))
        kb = 1.0 - math.exp(-blk_ms / 400.0)
        self._trk_base += kb * (self._trk_env - self._trk_base)

        rising = self._trk_env[chal] > 1.2 * self._trk_base[chal]
        clearly_stronger = self._trk_env[chal] > self._trk_env[cur] / margin_lin
        challenge = (unexplained[chal] > 0.3 * self._trk_env[cur]
                     and self._trk_env[chal] > 1e-4 and self._trk_voice_ref > 1e-3
                     and t_now > 0.05 and (rising or clearly_stronger))

        if t_now < self._trk_hold_until:
            self._trk_timer = 0
            state = "hold"
        elif challenge:
            self._trk_timer += 1
            state = "challenging"
            if self._trk_timer * blk_ms >= p["dwell_ms"]:
                retention = self._trk_env[cur] / max(self._trk_voice_ref, EPS)
                two_voice = retention > 0.6 and self._trk_wobble > 60.0
                # 能量迁移，或挑战峰明显更强（声部已漂移）且相位连续 -> 跟随
                glide = self._trk_wobble < 60.0 and (retention < 0.6 or clearly_stronger)
                if two_voice:
                    # 起音证据：旧声部能量仍在 + 相位差频摆动 = 独立新声部，保持旧中心
                    self._trk_hold_until = t_now + 0.30
                    if t_now - self._trk_last_hold_log > 0.5:
                        self.events.append({"t": t_now, "type": "hold_reonset",
                                            "detail": f"retention={retention:.2f} wobble={self._trk_wobble:.0f}Hz "
                                                      f"chal={self._trk_centers[chal]:.0f}Hz -> 保持旧中心"})
                        self._trk_last_hold_log = t_now
                    self._trk_timer = 0
                    state = "hold"
                elif glide:
                    # 能量迁移 + 相位连续 = 同一声部滑移，换轨跟随
                    self._commit_switch(chal, t_now, "glide_switch", retention)
                    state = "switch"
                else:
                    # 证据不足：检测窗尚未看全新峰，暂缓跟随
                    state = "defer"
        else:
            self._trk_timer = 0

        # 原声部丢失后的重新捕获
        cur = int(np.argmin(np.abs(self._trk_centers - self._trk_fc)))
        if self._trk_voice_ref > 1e-4 and self._trk_env[cur] < 0.3 * self._trk_voice_ref:
            self._trk_lost += 1
            if self._trk_lost * blk_ms >= p["lost_hold_ms"]:
                best = int(np.argmax(self._trk_env))
                if best != cur and self._trk_env[best] > 1e-4:
                    self._commit_switch(best, t_now, "reacquire", 0.0)
                self._trk_lost = 0
                self._trk_voice_ref = 0.0
        else:
            self._trk_lost = 0

        # 中心频率斜变
        k = 1.0 - math.exp(-blk_ms / max(p["f_slew_ms"], 1e-3))
        self._trk_fc += k * (self._trk_fc_target - self._trk_fc)
        return state

    def _commit_switch(self, band, t_now, ev_type, retention):
        # 抛物线插值得到亚带精度的峰频率
        m = int(np.clip(band, 1, self.N_TRACK_BANDS - 2))
        L = 20.0 * np.log10(self._trk_env[m - 1:m + 2] + EPS)
        denom = L[0] - 2.0 * L[1] + L[2]
        delta = 0.0 if abs(denom) < 1e-9 else float(np.clip(0.5 * (L[0] - L[2]) / denom, -1, 1))
        f_lo, f_hi = self._trk_centers[m - 1], self._trk_centers[m + 1]
        f_peak = self._trk_centers[m] * (f_hi / f_lo) ** (delta / 2.0)
        old = self._trk_fc_target
        self._trk_fc_target = _clamp(float(f_peak), F_MIN, NYQ_GUARD * self.fs)
        self._trk_timer = 0
        self._trk_voice_ref = 0.0
        self.events.append({"t": t_now, "type": ev_type,
                            "detail": f"{old:.0f}Hz -> {self._trk_fc_target:.0f}Hz "
                                      f"retention={retention:.2f} wobble={self._trk_wobble:.0f}Hz"})

    # ------------------------------------------------------------------ 处理
    def process(self, x):
        """单遍因果处理。返回 (y, taps)。已写出的样本不会被重算。"""
        x = np.asarray(x, dtype=np.float64)
        n = x.size
        p = self.params
        B = self.BLOCK
        nb = (n + B - 1) // B
        y = np.zeros(n)
        tap_det = np.zeros(n)
        tap_gain = np.zeros(n)
        tap_env = np.zeros(n)
        log = {k: [] for k in ("t", "gain_db", "env_db", "f_detect", "f_process",
                               "pole_r", "trk_state", "wobble")}
        band_envs = []
        ka_e = 1.0 - math.exp(-1.0 / (p["det_attack_ms"] * 1e-3 * self.fs))
        kr_e = 1.0 - math.exp(-1.0 / (p["det_release_ms"] * 1e-3 * self.fs))
        ka_g = 1.0 - math.exp(-1.0 / (p["attack_ms"] * 1e-3 * self.fs))
        kr_g = 1.0 - math.exp(-1.0 / (p["release_ms"] * 1e-3 * self.fs))
        ratio_gain = 1.0 - 1.0 / max(p["ratio"], 1.0)
        thr, max_cut = p["threshold_db"], p["max_cut_db"]
        mix = p["mix"]
        state_map = {"idle": 0, "challenging": 1, "defer": 2, "hold": 3, "switch": 4}

        env, gain_db = self._env, self._gain_db
        f_det, f_proc = self._f_det, self._f_proc
        kf = 1.0 - math.exp(-(B / self.fs) / (p["f_slew_ms"] * 1e-3))

        for bi in range(nb):
            i0, i1 = bi * B, min((bi + 1) * B, n)
            blk = x[i0:i1]
            t_now = i0 / self.fs

            trk_state = "idle"
            if p["tracker_enable"] > 0.5:
                trk_state = self._tracker_step(blk, t_now)
                f_det_target = self._trk_fc
            else:
                f_det_target = p["detect_freq"]
            f_det += kf * (f_det_target - f_det)
            f_proc += kf * (self._proc_target() - f_proc)

            # 检测通路（系数按实际签名缓存：f_det/q 不变即复用）
            b_d, a_d = self._cached_bandpass(f_det, p["detect_q"])
            det, self._zi_det = lfilter(b_d, a_d, blk, zi=self._zi_det)
            tap_det[i0:i1] = det

            # 包络 + 增益计算 + 增益平滑（逐样本）
            ad = np.abs(det)
            g_blk = np.empty(blk.size)
            e_blk = np.empty(blk.size)
            for i in range(blk.size):
                v = ad[i]
                env += (ka_e if v > env else kr_e) * (v - env)
                over = 20.0 * math.log10(env + 1e-9) - thr
                cut = over * ratio_gain if over > 0.0 else 0.0
                if cut > max_cut:
                    cut = max_cut
                tgt = -cut
                gain_db += (ka_g if tgt < gain_db else kr_g) * (tgt - gain_db)
                g_blk[i] = gain_db
                e_blk[i] = env
            tap_gain[i0:i1] = g_blk
            tap_env[i0:i1] = e_blk

            # 处理通路：只动这一个峰形滤波器的增益。
            # g/f/q/fs 任一变化才有新签名；稳态段直接复用上一块系数。
            b_p, a_p, r_pole, clamped = self._cached_peaking(
                float(g_blk[-1]), f_proc, p["process_q"])
            if clamped:
                self.pole_clamp_count += 1
            wet, self._zi_proc = lfilter(b_p, a_p, blk, zi=self._zi_proc)
            y[i0:i1] = mix * wet + (1.0 - mix) * blk

            log["t"].append(t_now)
            log["gain_db"].append(float(g_blk[-1]))
            log["env_db"].append(20.0 * math.log10(float(e_blk[-1]) + 1e-9))
            log["f_detect"].append(f_det)
            log["f_process"].append(f_proc)
            log["pole_r"].append(r_pole)
            log["trk_state"].append(state_map[trk_state])
            log["wobble"].append(self._trk_wobble)
            band_envs.append(self._trk_env.copy())

        self._env, self._gain_db = env, gain_db
        self._f_det, self._f_proc = f_det, f_proc
        self._sample_idx += n

        taps = {"detect": tap_det, "gain_db": tap_gain, "env": tap_env,
                "log": {k: np.array(v) for k, v in log.items()},
                "band_envs": np.array(band_envs),
                "band_centers": self._trk_centers.copy()}
        return y, taps


def model_response_db(g_db, f_center, q, f_list, fs):
    """RBJ 峰形滤波器在 f_list 处的理论幅频响应（dB）。用于核对实测与模型一致。"""
    b, a, _, _ = peaking_coeffs(g_db, f_center, q, fs)
    f_list = np.atleast_1d(np.asarray(f_list, dtype=float))
    w = 2.0 * np.pi * f_list / fs
    z1 = np.exp(-1j * w)
    z2 = np.exp(-2j * w)
    H = (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)
    return 20.0 * np.log10(np.abs(H) + EPS)
