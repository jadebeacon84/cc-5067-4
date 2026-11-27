# dyneq — 因果动态均衡（检测提速与缓存正确性）

一个逐样本包络、单遍因果的动态均衡内核，附带母带试听 Gradio 面板、冻结的
提速前参照、缓存键/轨迹交叉/检测共享的可执行探针，以及真实计时与音频证据。

- 处理块长 `DynamicEQ.BLOCK = 32`（48 kHz 下约 0.67 ms），13 个跟踪频带。
- 检测带通/处理峰形均为 RBJ 双二阶；包络与增益逐样本更新，因此 2 ms 短峰
  （约 96 样本、3 个块）也不会因“降低检测频率”而漏检。
- 所有“速度/增益/声音”结论都由脚本当次真实执行得到，写入
  `out/bench/detect_speed.json`，无固定数据、无伪报。

---

## 1. 图形入口（主要使用方式）

母带试听面板是本项目的主界面，所有音频与曲线都由真实 `dyneq.kernel.DynamicEQ`
实时计算。

```bash
# 仓库根目录；隔离模式 -I 隐含 -P，不会把当前目录加入 sys.path，
# 因此用会先插入仓库根的启动器：
/tmp/venv/bin/python -I scripts/panel.py

# 或者（不加 -I）：
/tmp/venv/bin/python -m dyneq.app
```

浏览器打开 <http://127.0.0.1:7860>。

面板含两个页签：

1. **完整参数运行**：由内核 schema 自动生成全部参数控件，输出干声/湿声/检测
   通路音频、动态增益+包络图、检测/处理中心轨迹、响应-模型对照，以及内核
   钳位/拒绝/跟踪事件的 JSON 反馈。
2. **检测频段 A/B 对照**：实际修改 `detect_freq`，对同一声音连续运行两个**真实**
   `DynamicEQ`（仅检测频段不同），叠加两次的动态增益与检测频率轨迹，提供
   干声、变更前/后湿声、逐样本差（后−前，原始刻度）试听与实测 JSON。
   `800 / 2000 / 3000 / 6000 Hz` 快捷按钮会把“变更后”频带填入并调用同一个
   真实对照回调（不是固定数据）。

> 说明：自动化环境无法代替人用耳朵听感；已做到的执行证据是——服务真实启动、
> `gradio_client` 经 HTTP 调用 `/_ab_compare` 与预设端点成功返回图/音频/JSON，
> 并用 `scripts/panel_check.py` 把每组 A/B 的 WAV/PNG/JSON 落到 `out/panel_ab/`
> 供人工试听。

无浏览器自检并生成 A/B 证据：

```bash
/tmp/venv/bin/python -I scripts/panel_check.py
# -> out/panel_ab/panel_ab.json, ab_*_A_before.wav, ab_*_B_after.wav,
#    ab_*_diff.wav, ab_compare.png
```

---

## 2. 命令行（仅作辅助）

```bash
# 旧样例全流程（7 节，P1–P7 + out/metrics.json + out/audio/*.wav）
/tmp/venv/bin/python -I scripts/run_all.py

# 检测计算提速基准 + 全部缓存/共享/短事件探针（P8/P9 + JSON + speedup_*.wav）
/tmp/venv/bin/python -I scripts/bench_detect_speed.py

# 测试（28 项：内核等价性、缓存键、交叉、别名、共享失效、短事件等）
/tmp/venv/bin/python -I -m pytest tests/ -q
```

冻结的提速前参照是 `dyneq/ref_baseline.py`，它是提速前 `HEAD:dyneq/kernel.py`
的逐字节副本（仅加一行出处注释），作为“动态均衡的运行参照”，禁止为通过测试
而修改。基准与等价性测试同时驱动 `dyneq.ref_baseline.DynamicEQ` 与
`dyneq.kernel.DynamicEQ`。

---

## 3. 检测计算提速：计算顺序与缓存规则

### 3.1 每块的真实计算顺序

`DynamicEQ.process` 对每个 32 样本块按以下顺序执行（顺序本身不变，保证与冻结
基线逐位一致）：

1. **（可选）跟踪器** `_tracker_step`：13 个复数谐振器 `lfilter` + 包络判决。
   谐振器系数只取决于中心频率与极点半径 `r`，因此在 `_tracker_reset` 里
   **预计算一次** `_trk_res_b / _trk_res_a[m]`，不再每块为 13 个频带重做
   三角与 `np.array([...])` 构造；`r` 仍用于裙边/能量公式。
2. **频率平滑**：检测/处理中心按 slew 向目标逼近。
3. **检测带通系数**：`_cached_bandpass(f_det, detect_q)`。签名
   `(f, Q, fs)` 不变就复用上一块 `(b, a)`，否则调用纯函数
   `bandpass_coeffs` 重算。
4. **检测 `lfilter`**：保留并推进 `zi_det`。
5. **逐样本包络与增益**：在块内 32 个样本上逐样本做 attack/release 包络和
   增益计算（**不降低检测/包络更新率**）。
6. **处理峰形系数**：块末用该块最后一个增益值
   `_cached_peaking(g_db, f_proc, process_q)`。签名 `(g, f, Q, fs)` 任一变化
   才重算 `peaking_coeffs`（含极点安全钳位判定）。
7. **处理 `lfilter`**：保留并推进 `zi_proc`，输出湿声。

`CoeffCache` 是**每类一个最近签名槽**的有界备忘录（不是无限字典）：参数来回
交替时最多多一次未命中并重算，绝不会返回错误系数，长跑也不累积条目。命中
返回的系数与直接调用工厂函数逐位相同——这一点由与冻结基线的逐位相等测试
锁定。

### 3.2 为什么“按频带编号做缓存键”是错的

频带**编号**是可变标签：同一条自动化曲线在不同块可能落到不同编号，两条节点
的编号也可能在交点后“易主”。若缓存只认编号，交点之后会拿到另一条轨迹的旧
系数。

可执行场景（`dyneq/speedup.py: make_crossing_trajectories`）：

- 两个检测节点 A/B **相同起始频率**，不同自动化曲线；
- 用 128 样本的教学块，两条直线在**块正中央**精确相交
  （交叉样本 `11584 = 90*128 + 64`，`t_cross = 0.241333 s`），块首频率
  `2700 / 1700 Hz`，块末互换；激励为 `2700 Hz` 纯音。
- 对照三种键：逐样本更新（live）、含**轨迹版本**的键 `("bp", traj_version,
  f, Q, fs)`、以及错误的“按当前频带编号”键。

实测（`out/bench/detect_speed.json -> crossing_probe`，确定性结果）：

- 轨迹版本缓存与逐样本更新的湿声/增益**逐位相同**；
- 编号键相对版本键，B 节点交点后**第一段错误衰减**起点恰为 `0.241333 s`
  （紧贴交点），持续 23 个样本，峰值误差约 **2.00 dB**、均值约 1.56 dB；
  相对“稳定持块”路径峰值误差更大（>3 dB 判据下同样定位到交点后首段）；
- 编号缓存记录到的最大频率错配 `max_freq_mismatch_hz > 200`。

### 3.3 仅换显示名称的别名对照

显示名称**不进入签名/版本**。构造一条与 A 轨迹完全相同、版本号相同、仅
`name/display_name_only` 不同的节点再跑同一个 `VersionCoeffCache`：

- 重命名后**新未命中 = 0**、新增命中 19200；
- 两段湿声 `array_equal`，`wet_max_abs_diff = 0.0`。

证据音频：`out/audio/speedup_alias_{original,renamed,diff}.wav`
（diff 为数字零）。

### 3.4 频带重叠时共享检测 + 一次频移使部分共享失效

`SharedDetectionBank` 以完整键 `(f, Q, fs, version)` 记忆检测 `lfilter`：

- 同一块内相同键的多个请求只跑一次 `lfilter`，其余直接分发缓存的 `last_y`；
- 自动化频移改变某节点的键时，只让**这一个键**失效；失效前先
  `split_state` 把边界 `zi` 迁移到新键（保持滤波器状态连续），且迁移必须在
  处理频移块**之前**完成。

`overlap_sharing_scenario` 的全量核验（确定性结果）：

- `max_det_diff = max_gain_diff = 0.0`，`wet_bitexact = true`
  （共享路径 vs 每节点独立检测）；
- 1.2 s 场景：3600 次块请求中仅 **2850 次 `lfilter`**、750 次共享命中，
  频移后共 2 个唯一键；
- 频移前后在 `3000 / 1800 Hz` 的衰减轨迹由全样本测量核验
  （见 JSON `overlap_sharing`）。

多节点扩展的是**检测计算**本身，因此收益远大于单内核系数备忘录（真实计时，
见第 4 节）：N=2 ≈ **1.75×**、N=4 ≈ **3.1×**、N=8 ≈ **5.2×**。

### 3.5 不能靠降低检测频率忽略短事件

`short_burst_train` 含 **57** 个 2 ms 短峰。刻意轮询检测块得到：

| 检测更新策略 | 命中短峰 |
| --- | --- |
| 每块（poll=1） | 57 |
| 每 4 块 | 57 |
| 每 8 块 | 39 |
| 每 16 块 | 20 |
| **本内核逐样本包络** | **57** |

因此提速只来自消除重复系数构造与共享相同检测，**不通过降低检测/包络更新率**。
冻结基线与优化内核在该负载上输出与增益轨迹逐位相同
（`gain_trajectory_max_diff_db = 0.0`）。

---

## 4. 真实速度与成本分解（本机当次执行）

权威数字以提交的 `out/bench/detect_speed.json` 为准（含 Python/NumPy/平台与
git HEAD）。计时采用**基线/优化交替配对**取中位数，以降低调度与热漂移偏差；
下列为本仓库提交当次的数量级（容器内绝对 µs 会随机器波动）：

| 负载 | 冻结基线 µs/块 | 优化 µs/块 | 顺序中位比 | 交替配对比 | 与基线等价 |
| --- | ---: | ---: | ---: | ---: | --- |
| `dynamic_pilot`（稳定检测频带，4 s 导频栅格） | 18.12 | 17.67 | 1.025× | 1.025× | 输出/增益逐位相同 |
| `tracker_glide`（13 频带跟踪器 + 滑音） | 118.64 | 112.32 | 1.056× | 1.052× | 输出/增益逐位相同 |
| `continuous_param_sweep`（每 50 ms 改 f/Q） | 18.33 | 18.74 | 0.978× | 0.974× | 该负载未做逐位断言 |

同次运行的缓存统计解释了差异：`dynamic_pilot` 总命中率约 **75%**（检测带通
6000 次查找仅 1 次未命中；峰形增益逐块变化故峰形系数多为未命中），
`tracker_glide` 约 **38%**，而极端连续扫描仅约 **1.8%**（110 命中 / 5890 未命中）。

多节点共享 `lfilter` 的真实计时（同份 JSON `sharing_scaling`）：N=2
独立 4.44 → 共享 2.54 µs/块（**1.75×**），N=4 8.53 → 2.75（**3.11×**），
N=8 16.69 → 3.21（**5.20×**）。

成本分解（cProfile，跟踪器负载，`cost_decomposition_tracker.pct_by_category`）：

- `lfilter/scipy` ≈ **25%**
- `_tracker_step` Python 判决 ≈ **12%**
- `process` 包络/增益循环 ≈ **8%**
- 系数工厂函数本身仅 ≈ **0.3–0.5%**
- 其余 ≈ **54%**

**诚实解读**：系数工厂在总耗时中占比很小，所以单内核缓存只在“参数稳定/平滑”
素材上带来几个百分点的收益（跟踪器谐振系数预计算是其中较稳定的来源）；在
“连续调参 + 增益逐块变化”的极端扫描负载下，签名几乎块块变、命中率只有约
1.8%，缓存簿记带来约 2–3% 的开销（配对中位 **0.974×**，即略慢）。这是**每块成本**
指标内的中性/轻微负面结果，按要求如实保留，不做美化。缓存的主要价值是：
(1) 明确“按实际参数签名失效”的正确语义并支撑轨迹版本/别名论证；(2) 严格有界；
(3) 在真正重叠的多检测节点场景通过共享 `lfilter` 获得 1.7–5.1× 的实质提速。

**撤回判据复核**：任务要求“若频段响应及每块成本之外的行为退化，则撤回相应
优化”。全部 28 个测试通过，优化内核与冻结 HEAD 在动态素材、跟踪器素材、
分块自动化、短事件上输出/增益/检测轨迹**逐位相同**；起释时间常数
（attack ≈ 58 ms、release ≈ 422 ms）、最大极点半径
（≈ 0.9381295）、钳位次数 0、输出有限性在基线/优化间完全一致；导频栅格频段
响应模型误差与既有测试容限一致。除“每块成本”这一被度量项外没有行为退化，
故保留优化。

---

## 5. 证据地图

| 结论 | 执行/测试 | 落盘证据 |
| --- | --- | --- |
| 优化内核与 HEAD 基线逐位一致 | `tests/test_detect_speedup.py` | `dyneq/ref_baseline.py`；JSON `kernel_speed.*_bitexact_*` |
| 稳定参数命中缓存 | `test_coeff_cache_reuses_when_params_stable` | JSON `kernel_speed.*.coeff_cache` |
| 交点精确定位 + 首段错误衰减 | `test_crossing_*`, `test_index_cache_first_bad_at_crossing` | JSON `crossing_probe`；`out/figs/P9_crossing_cache.png` |
| 版本键=逐样本、别名零新未命中且湿声零差 | `test_version_cache_*`, `test_alias_*` | `speedup_cross_*`、`speedup_alias_*` |
| 共享检测零差 + 频移部分失效 | `test_shared_detection_*` | JSON `overlap_sharing`/`sharing_scaling`；`speedup_overlap_wet_shared.wav`；P8 |
| 不靠降检测率漏短事件 | `test_polled_*`, `test_optimized_kernel_*` | JSON `short_events`；P8 |
| 起释/极点稳定基线一致 | 全套测试 + bench 第 6 节 | JSON `attack_release_stability` |
| 频段响应实测 vs RBJ 模型 | bench 第 7 节 | JSON `band_response`；`out/figs/P8_detect_speedup.png` |
| 每块成本 / 成本分解 | bench 第 1–2 节 | JSON `kernel_speed`、`cost_decomposition_tracker`；P8 |
| 旧样例端到端不回归 | `scripts/run_all.py`（退出 0） | `out/metrics.json`，`out/figs/P1..P7_*.png` |
| 面板真实 A/B 与 HTTP 调用 | `scripts/panel_check.py` + `gradio_client` 实测 | `out/panel_ab/panel_ab.json` 与 WAV/PNG |

主要图：`out/figs/P8_detect_speedup.png`（每块成本、成本分解、共享扩展、
交叉增益、短事件、频段响应）与 `out/figs/P9_crossing_cache.png`（交叉轨迹与
湿声差）。

---

## 6. 环境备注

仓库使用 `/tmp/venv`（Python 3.11，NumPy/SciPy/Matplotlib/pytest/Gradio）。
运行读取外部数据的脚本时按环境约定使用 `-I`；注意 `-I` 隐含 `-P`，因此图形
入口在隔离模式下请使用 `scripts/panel.py` 启动器。
