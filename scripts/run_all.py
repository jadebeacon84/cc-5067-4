"""一键生成动态均衡全部交付物：真实音频、响应图、指标。

用法: python scripts/run_all.py  (产物写入 out/)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from dyneq import stimuli, analysis
from dyneq.kernel import DynamicEQ, model_response_db, R_MAX
from dyneq.render import render, save_wav, pilot_gain_db, measure_time_constants

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
AUDIO = os.path.join(OUT, "audio")
FIGS = os.path.join(OUT, "figs")
FS = 48000

TRACK_PARAMS = dict(detect_freq=2000, tracker_enable=1, process_follow=1,
                    threshold_db=-18, ratio=4, max_cut_db=15, dwell_ms=60)
DYN_PARAMS = dict(detect_freq=3000, detect_q=4, process_freq=3000, process_q=3,
                  threshold_db=-30, ratio=3, max_cut_db=18,
                  attack_ms=10, release_ms=180)


def render_tracker_pair():
    xg, mg = stimuli.glide_tone(FS)
    xr, mr = stimuli.reonset_tone(FS)
    yg, tg, kg = render(xg, FS, **TRACK_PARAMS)
    yr, tr, kr = render(xr, FS, **TRACK_PARAMS)
    return ({"x": xg, "y": yg, "taps": tg, "kernel": kg, "meta": mg},
            {"x": xr, "y": yr, "taps": tr, "kernel": kr, "meta": mr})


def save_taps_wavs(tag, res):
    x, y, taps = res["x"], res["y"], res["taps"]
    save_wav(os.path.join(AUDIO, f"{tag}_dry.wav"), x, FS)
    save_wav(os.path.join(AUDIO, f"{tag}_wet.wav"), y, FS)
    save_wav(os.path.join(AUDIO, f"{tag}_detect.wav"), taps["detect"], FS)
    gain_audio = np.clip(taps["gain_db"] / 30.0, -1.0, 0.0)   # 控制信号映射到 [-1,0]
    save_wav(os.path.join(AUDIO, f"{tag}_gain.wav"), gain_audio, FS)
    env_audio = np.clip((20 * np.log10(taps["env"] + 1e-9) + 60.0) / 60.0, -1.0, 1.0)
    save_wav(os.path.join(AUDIO, f"{tag}_envelope.wav"), env_audio, FS)


def main():
    os.makedirs(AUDIO, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    metrics = {}

    # ---------- 1. 双音：滑移 vs 重新起振 ----------
    print("[1/7] 渲染滑移/重新起振双音 ...")
    res_g, res_r = render_tracker_pair()
    save_taps_wavs("glide", res_g)
    save_taps_wavs("reonset", res_r)

    log_g, log_r = res_g["taps"]["log"], res_r["taps"]["log"]
    t = log_g["t"]
    ideal_g = analysis.ideal_freq_glide(t, 2000, 2600, 2.0, 3.0)
    ideal_r = analysis.ideal_freq_reonset(t, 2000, 2600, 4.65)
    metrics["glide"] = {
        "final_f_process_hz": float(log_g["f_process"][-1]),
        "crossing_lag_s": analysis.crossing_lag(t, log_g["f_process"], ideal_g, 2300.0),
        "misduck_energy_dbs": analysis.misduck_energy(t, log_g["gain_db"],
                                                      log_g["f_process"], ideal_g),
        "events": [(round(e["t"], 3), e["type"]) for e in res_g["kernel"].events
                   if e["type"] != "tracker_reset"],
    }
    metrics["reonset"] = {
        "f_process_at_4p4s_hz": float(log_r["f_process"][np.searchsorted(log_r["t"], 4.4)]),
        "final_f_process_hz": float(log_r["f_process"][-1]),
        "switch_delay_s": analysis.tracking_delay(t, log_r["f_process"], 2600, 4.65),
        "misduck_energy_dbs": analysis.misduck_energy(t, log_r["gain_db"],
                                                      log_r["f_process"], ideal_r),
        "events": [(round(e["t"], 3), e["type"]) for e in res_r["kernel"].events
                   if e["type"] != "tracker_reset"],
    }
    analysis.plot_tracker(res_g, res_r, res_g["meta"], res_r["meta"],
                          os.path.join(FIGS, "P1_tracker.png"))

    # 检测包络相等性（标定同款谐振器 + 60ms 驻留窗平滑）
    eg = stimuli._arrival_envelope(res_g["x"], FS, 2600.0)
    er = stimuli._arrival_envelope(res_r["x"], FS, 2600.0)
    tt = np.arange(eg.size) / FS
    m = (tt >= 2.1) & (tt <= 3.4)
    ksm = int(0.06 * FS)
    kern = np.ones(ksm) / ksm
    dev = 20 * np.abs(np.log10((np.convolve(eg[m], kern, "same") + 1e-9) /
                               (np.convolve(er[m], kern, "same") + 1e-9)))
    metrics["equal_envelope"] = {"max_dev_db": float(dev.max()),
                                 "mean_dev_db": float(dev.mean())}

    # ---------- 2. 检测信号/处理响应/总输出分轨观察 ----------
    print("[2/7] 分轨观察图 ...")
    i0, i1 = int(1.9 * FS), int(3.3 * FS)
    t_seg = np.arange(i0, i1) / FS
    analysis.plot_taps(t_seg, res_g["taps"]["detect"][i0:i1],
                       20 * np.log10(res_g["taps"]["env"][i0:i1] + 1e-9),
                       res_g["taps"]["gain_db"][i0:i1], res_g["y"][i0:i1],
                       os.path.join(FIGS, "P6_taps.png"))

    # ---------- 3. 固定音 + 可变窄带共振：时间常数与目标衰减 ----------
    print("[3/7] 固定音+可变共振 ...")
    xf, mf = stimuli.fixed_plus_resonance(FS)
    yf, tf, kf = render(xf, FS, **DYN_PARAMS)
    save_taps_wavs("fixed_resonance", {"x": xf, "y": yf, "taps": tf})
    logf = tf["log"]
    steady = {}
    for (t0, t1, lvl) in mf["bursts"]:
        msk = (logf["t"] >= t0 + 0.4) & (logf["t"] <= t1 - 0.05)
        steady[f"{t0}-{t1}s"] = float(np.mean(logf["gain_db"][msk]))
    cut2 = -steady["3.0-4.0s"]
    base_msk = (logf["t"] >= 2.3) & (logf["t"] <= 2.9)
    baseline = float(np.mean(logf["gain_db"][base_msk]))
    tc = measure_time_constants(logf["t"], logf["gain_db"], 3.0, 4.0, cut2, baseline)
    # 目标频段实际衰减（3kHz 带内 RMS 比）
    from dyneq.kernel import bandpass_coeffs
    from scipy.signal import lfilter
    b, a = bandpass_coeffs(3000, 4.0, FS)
    i0, i1 = int(3.4 * FS), int(3.95 * FS)
    rms_d = np.sqrt(np.mean(lfilter(b, a, xf)[i0:i1] ** 2))
    rms_w = np.sqrt(np.mean(lfilter(b, a, yf)[i0:i1] ** 2))
    atten_meas = 20 * np.log10(rms_w / rms_d)
    # 邻近/远场导频受影响
    pilots_nb = {}
    for f in (500.0, 997.0, 2400.0, 3600.0):
        pilots_nb[f] = pilot_gain_db(xf, yf, f, FS, 3.4, 3.95)
    g_steady = steady["3.0-4.0s"]
    model_nb = {f: float(model_response_db(g_steady, 3000, 3, [f], FS)[0])
                for f in pilots_nb}
    metrics["fixed_resonance"] = {
        "steady_cut_db": steady,
        "baseline_db": baseline,
        "attack_ms": tc["attack_ms"], "release_ms": tc["release_ms"],
        "attack_cfg_ms": 10.0, "release_cfg_ms": 180.0,
        "band_energy_attenuation_db": float(atten_meas),
        "note_band": "带内能量衰减含激励带宽与滤波器裙边的频率加权，纯音目标衰减见 pilot_grid 3000Hz",
        "gain_path_cut_db": g_steady,
        "pilot_gain_db": pilots_nb, "model_gain_db": model_nb,
    }
    analysis.plot_timeconst(logf["t"], logf["gain_db"], mf["bursts"], tc,
                            10.0, 180.0, os.path.join(FIGS, "P3_timeconst.png"))

    # ---------- 4. 导频栅格：非目标频率响应 vs 模型 ----------
    print("[4/7] 导频栅格响应对照 ...")
    xp, mp = stimuli.pilot_grid(FS)
    yp, tp, kp = render(xp, FS, **DYN_PARAMS)
    save_wav(os.path.join(AUDIO, "pilot_grid_dry.wav"), xp, FS)
    save_wav(os.path.join(AUDIO, "pilot_grid_wet.wav"), yp, FS)
    resp = analysis.measure_response(xp, yp, mp["freqs"], FS, mp["windows"], tp)
    metrics["pilot_grid"] = [{
        "window": r["window"], "gain_db": r["gain_db"],
        "max_model_err_db": r["max_err"],
        "meas_db": [float(v) for v in r["meas"]],
        "model_db": [float(v) for v in r["model"]],
    } for r in resp]
    f_model = np.geomspace(100, 16000, 600)
    analysis.plot_response(
        mp["freqs"], resp[0]["meas"], resp[1]["meas"], f_model,
        model_response_db(resp[0]["gain_db"], resp[0]["f_process"], 3, f_model, FS),
        model_response_db(resp[1]["gain_db"], resp[1]["f_process"], 3, f_model, FS),
        resp[0]["gain_db"], resp[1]["gain_db"],
        os.path.join(FIGS, "P2_response.png"))

    # ---------- 5. 跟踪速度 vs 误压能量权衡 ----------
    print("[5/7] 驻留时间权衡扫描 ...")
    dwells = [0, 20, 60, 120, 240]
    delay_g, mis_g, delay_r, mis_r = [], [], [], []
    for d in dwells:
        prm = dict(TRACK_PARAMS, dwell_ms=d)
        yg2, tg2, _ = render(res_g["x"], FS, **prm)
        yr2, tr2, _ = render(res_r["x"], FS, **prm)
        lg, lr = tg2["log"], tr2["log"]
        delay_g.append(analysis.crossing_lag(t, lg["f_process"], ideal_g, 2300.0))
        mis_g.append(analysis.misduck_energy(t, lg["gain_db"], lg["f_process"], ideal_g))
        delay_r.append(analysis.tracking_delay(t, lr["f_process"], 2600, 4.65))
        mis_r.append(analysis.misduck_energy(t, lr["gain_db"], lr["f_process"], ideal_r))
    metrics["tradeoff"] = {"dwell_ms": dwells, "delay_glide_s": delay_g,
                           "misduck_glide_dbs": mis_g, "delay_reonset_s": delay_r,
                           "misduck_reonset_dbs": mis_r}
    analysis.plot_tradeoff(dwells, delay_g, mis_g, delay_r, mis_r,
                           os.path.join(FIGS, "P4_tradeoff.png"))

    # ---------- 6. 稳定性：连续自动化扫描 + 非法值注入 ----------
    print("[6/7] 稳定性/钳位验证 ...")
    k = DynamicEQ(FS)
    k.set_params(threshold_db=-60, ratio=6, max_cut_db=24, attack_ms=2,
                 release_ms=60, tracker_enable=0)
    n_chunks = 40
    chunk = int(0.1 * FS)
    rng = np.random.default_rng(3)
    t_sub, f_sub, f_applied, q_applied = [], [], [], []
    pole_r_all, gain_all = [], []
    y_all = []
    feedbacks = []
    for i in range(n_chunks):
        f_cmd = 20.0 * (21600.0 / 20.0) ** (i / (n_chunks - 1))
        q_cmd = 0.2 + (24.0 - 0.2) * i / (n_chunks - 1)
        if i == 10:
            f_cmd = 5.0            # 非法：低于下限
        if i == 20:
            f_cmd = 30000.0        # 非法：高于奈奎斯特保护
        if i == 15:
            q_cmd = 0.0            # 非法 Q
        if i == 25:
            q_cmd = 1e6            # 非法 Q
        if i == 30:
            f_cmd = float("nan")   # 非法：拒绝
        res = k.set_params(process_freq=f_cmd, process_q=q_cmd,
                           detect_freq=min(f_cmd, 8000.0))
        for name, st, msg in res:
            if st != "ok":
                feedbacks.append(f"chunk{i} {name}: [{st}] {msg}")
        t_sub.append(i * 0.1)
        f_sub.append(f_cmd if np.isfinite(f_cmd) else -1)
        f_applied.append(k.params["process_freq"])
        q_applied.append(k.params["process_q"])
        x = (0.15 * np.sin(2 * np.pi * 200 * np.arange(chunk) / FS)
             + 0.15 * np.sin(2 * np.pi * 1000 * np.arange(chunk) / FS)
             + 0.1 * np.sin(2 * np.pi * 5000 * np.arange(chunk) / FS)
             + 0.05 * rng.standard_normal(chunk))
        y, taps = k.process(x)
        y_all.append(y)
        pole_r_all.append(taps["log"]["pole_r"])
        gain_all.append(taps["log"]["gain_db"])
    y_all = np.concatenate(y_all)
    pole_r_all = np.concatenate(pole_r_all)
    t_log = np.arange(pole_r_all.size) * 32 / FS
    metrics["stability"] = {
        "pole_r_max": float(pole_r_all.max()),
        "pole_clamp_events": k.pole_clamp_count,
        "output_finite": bool(np.all(np.isfinite(y_all))),
        "output_peak": float(np.max(np.abs(y_all))),
        "illegal_feedback": feedbacks,
        "f_applied_range": [float(min(f_applied)), float(max(f_applied))],
        "q_applied_range": [float(min(q_applied)), float(max(q_applied))],
    }
    analysis.plot_stability(t_log, pole_r_all, t_sub, f_sub,
                            np.interp(t_log, t_sub, f_applied),
                            FS, os.path.join(FIGS, "P5_stability.png"))

    # ---------- 7. 因果性（录音安全）：前缀一致 ----------
    print("[7/7] 因果性验证 ...")
    xp2, _ = stimuli.pilot_grid(FS, dur=2.0)
    y_full, _, _ = render(xp2, FS, **DYN_PARAMS)
    n_pre = int(1.3 * FS)
    y_pre, _, _ = render(xp2[:n_pre], FS, **DYN_PARAMS)
    causal_ok = bool(np.array_equal(y_full[:n_pre], y_pre))
    metrics["causality"] = {"prefix_samples": n_pre, "bit_exact": causal_ok}

    # 频谱图（干声 vs 输出，无响度归一化）
    analysis.plot_spectrograms(
        [("滑移", res_g["x"], res_g["y"]), ("重新起振", res_r["x"], res_r["y"])],
        FS, os.path.join(FIGS, "P7_spectrograms.png"))

    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("完成。指标摘要：")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k in ("glide", "reonset", "equal_envelope",
                               "fixed_resonance", "stability", "causality")},
                     ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
