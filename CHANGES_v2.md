# dataset-generation-v2 变更记录

基线：`TUTUisAbigCAT/dataset-generation` commit `ce31920`（2026-11-04）。修改依据见 [docs/代码修改指南.md](docs/代码修改指南.md)。

**兼容性原则**：所有新功能默认关闭。不修改 `dataset_config.json`，`training_ema.decay = 0` 时训练与旧版逐位一致（有测试验证）。`TSImputers/S4Model.py`、`signal_normalization.py` 和上游快照没有改动，旧 checkpoint 的源码身份校验不受影响。

## 修复：原仓库测试套件中已有的 14 个失败

在基线代码上运行 `python -B -m unittest discover -s tests`（CPU 版 torch 2.3.1，与集群运行记录的版本一致），结果为 630 个测试中 14 个报错，原因都是测试没有随代码更新：

| 测试 | 原因 | 修复 |
|---|---|---|
| `test_dreamt_loader` 中 8 个 | 测试读取了仓库里提交的集群配置（88 人评估协议、minmax 范围），与 4 人测试数据冲突 | 在 `PreprocessorTests.setUp` 中通过 `DREAMT_CONFIG_PATH` 指向空配置，使测试与仓库配置隔离 |
| `test_early_stopping` 3 个、`test_window_config` 2 个 | `run_simulation.py` 新增了 S4 dropout 检查，但测试用的小模型里没有 S4 层 | `TinyTestModel` 增加 8 个不参与前向计算的 `S4Layer` 占位模块 |
| `test_masked_denoising` 1 个 | 用"第一个 `report[...].update` 分支"定位代码片段，后来的提交在前面插入了 `if in_sample_evaluation:` 分支 | 按 `observed_masks` 条件精确定位 |

## P0-1：整夜生成的缺失标记阈值

- `night_data.save_night_npz(..., missing_threshold_seconds=None)`：默认行为不变，遇到负值仍然报错。给定阈值（不超过 0.3 s，即 200 bpm）时，低于阈值的生成值写为 DREAMT 缺失标记 0，并在返回的报告中记录 `written_as_missing_samples` 和 `generated_negative_samples`。
- `scripts/generate_night.py --missing-threshold-seconds 0.2`。启用时报告中的 `invalid_values_clipped_or_repaired` 如实记为 `true`。

## P0-2：评估用 EMA 与逐轮 checkpoint

- 新增 `training_ema.py`：配置项 `training_ema.decay`，默认 0 表示关闭。EMA 只用于每轮验证和验证器保存的 checkpoint；`saved_models/<dataset>/model.pth` 仍是训练中的实时权重。
- `training_wavestitch.py` 在 `scripts/verify_upstream_boundary.py` 的白名单中新增 4 处精确插入（`EMA_EDITS`），审计脚本照常通过。新增 `remove_training_ema_edits`；`remove_training_input_dropout_edits` 同时接受新版、旧版 dropout 适配器和更早的源码，因此旧运行的源码对比仍然可用。
- `epoch_validation.py`：新增 `validation.save_epoch_checkpoints`，保存 `validation/epoch_NNNN/model.pth`，并在 `metrics.json` 中记录路径和 SHA256。
- `run_simulation.py` 在运行记录中写入 EMA 的配置和最终状态；`training_ema.py` 已加入 `run_dreamt` 的源码快照、sweep 的哈希清单和源码检查清单。

## P1：评估与实验设计

- `compare_s092_sweep_best.py --checkpoint-epoch N`：冻结并采样任意一个已保存的 epoch（配合 `--seed` 做多 seed 复评）。逐轮 checkpoint 不存在或 SHA256 不匹配时直接拒绝，不会回退到最优 checkpoint。样本汇总中的 `checkpoint_kind` 会如实标注为 `epoch_NNNN`。
- 新增 `ibi_structure_metrics.py`：25 Hz 阶梯（平台）结构、配对 Wasserstein-1，以及可选的 `to_sample_and_hold` 后处理。每轮写入 `metrics.json` 的 `diagnostics.ibi_structure`；不参与选模型，`epoch_metrics.csv` 的列不变。
- 新增 `scripts/make_holdout_split.py`：生成 S092 只在验证集中的划分和对应配置，scaler 不会使用 S092。
- 新增 `configs/dreamt_recommended.json`：EMA 0.9999、`masked` 缺失值策略、S4 dropout 0.03、关闭输入 dropout、保存逐轮 checkpoint。
- Trillium 作业脚本：`submit_v2_gpu_tests.sh`（把代码复制到 `$SLURM_TMPDIR` 后跑完整测试，因为作业内 `$PROJECT` 只读）和 `submit_v2_recommended.sh`（推荐配置，默认 50 个 epoch，可用 `EPOCHS` 覆盖；Trillium 单作业最长 24 小时且训练不支持续训）。两者都显式加载 `scipy-stack/2024a`，与以往运行的 numpy 1.26.4 环境一致。

## P2：维护

- README 统一为 Python 3.11 和 `.venv`，与 `submit*.sh` 及运行记录一致；`requirements_cuda.txt` 固定为运行记录中的 torch 2.3.1，`requirement.txt` 改为引用它，不再重复内容。
- `select_validation_windows` 新增 `preferred_start_sample` / `preferred_start_seconds` 两个名称正确的键，旧键 `preferred_second_start` 保留以兼容已有记录。

## 有意未做

- 输入 dropout 的"排除被擦除点的 loss"方案（推荐配置直接关闭输入 dropout）。
- `type=bool` 命令行参数、`model.pth` 改名、路径外置：会改动白名单或路径约定，收益较低。
- 不改 `S4Model.py` 中的 `Dropout2d`，不删除重复的 sweep 配置（对比脚本按 12 组配置校验；已确认配置 0 与 3 逐位相同）。
- `compare_s092_sweep_best.py` 的必需源码清单不加入新文件，否则旧 sweep 的冻结快照无法再对比。

## 验证

- `python -B -m unittest discover -s tests`：654 个测试全部通过，跳过 20 个（全部需要真实 GPU）。基线为 630 个测试、14 个报错。
- `python scripts/verify_upstream_boundary.py`：通过，`training_ema_edits` 共 4 处，撤销允许的编辑后训练循环与上游一致。
- 以上验证使用 CPU 版 torch 2.3.1（conda-forge）。需要 GPU 的测试和真实训练请在 Fir 的 GPU 节点上运行：`python -B -m unittest discover -s tests -v` 和 `python scripts/check_cuda.py`。

## v2.1：按《诊断与修改指南（精简版）》的修改

依据 [docs/诊断与修改指南_精简版.md](docs/诊断与修改指南_精简版.md)。范围：原始数据不改；归一化沿用 [−1,1]；缺失值策略沿用 `preserve`；暂不处理随机种子问题。所有新选项默认关闭，默认配置下训练与之前逐位一致。

### 评估

- **任意受试者评估**：新增 `scripts/evaluate_subject_windows.py` 和共享模块 `subject_windows.py`。对任意配置中的受试者，按 30/60/90/120 s 步长取 2 分钟窗口，逐窗口独立生成并与真实 IBI 比较；输出可直接用 `plot_s092_all_windows_kde.py --folder subject_<SID>` 画图，并汇总到 `subjects_summary.csv`。`night_data.prepare_night` 新增 `allow_time_gaps`（默认 False），评估时跨越时间空缺的窗口会被排除。
- **指标扩充**：新增 `distribution_metrics.py`（合并 W1、KS、按睡眠阶段的 W1、均值偏差、标准差比、逐窗口 KL 的中位数 / 90% 分位数 / 超过 1 的比例、composite）和 `hrv_metrics.py`（从 25 Hz 序列读出逐拍间期，计算 RMSSD、SDNN、pNN50、平均心率）。每轮验证写入 `metrics.json` 的 `diagnostics.pooled_distribution`、`window_kl`、`hrv`；`epoch_metrics.csv` 末尾新增 `pooled_w1_ms`、`pooled_ks`、`window_kl_median` 三列。
- **可配置的挑选指标**：`validation.selection_metric` 可选 `kl`（默认，不写入配置）、`w1`、`ks`、`composite`。非 `kl` 时额外保存 `best_<metric>_model.pth` 并生成对应的 manifest；`best_kl_model.pth` 和早停逻辑不变。
- **条件有效性验证**：新增 `scripts/check_condition_effect.py`，四项检查：真实数据中条件能解释的比例（R²）、换睡眠阶段、打乱阶段条件（按真实阶段对齐的 W1）、互换人口学信息的距离矩阵；按指南中的阈值给出判定，并输出 `condition_checks.png`。

### 训练

- **学习率调度与梯度裁剪**：新增 `training_schedule.py`，配置块 `training_schedule`（`warmup_steps`、`decay`: none/cosine、`min_lr_ratio`、`clip_grad_norm`）。`training_wavestitch.py` 新增 5 处白名单编辑（`TRAINING_SCHEDULE_EDITS`）：`optimizer.step()` 这一行原文不变，前面加梯度裁剪，后面更新学习率；`model.pth` 保存之后额外保存 `model_ema.pth`。新增 `remove_training_schedule_edits`，旧的反向函数照常可用。
- **模型尺寸可配置**：新增 `model_config.py`，配置块 `model`（`num_res_layers`、`res_channels`、`skip_channels`、`s4_dstate`）。没有该配置块时训练命令与之前完全相同；`run_simulation.py` 会核对配置是否真正传入训练，`night_checkpoint.load_inference_config` 也按该配置块构建推理模型。
- **EMA 最终权重**：`TrainingEMA.save_evaluation_weights` 在训练结束时写出 `saved_models/<dataset>/model_ema.pth`（EMA 关闭时与 `model.pth` 相同），运行报告记录其路径和 SHA256。
- 新模块已登记到 `run_dreamt.py` 的源码快照、`run_all88_s092_sweep.py` 的 `SNAPSHOT_SOURCES` 和 `run_simulation.py` 的源码哈希清单。

### 配置与作业

- `configs/dreamt_recommended.json`：缺失值策略改回 `preserve`；保持 [−1,1]、S4 dropout 0.03、EMA 0.9999、逐轮 checkpoint。
- 新增 `configs/dreamt_b1.json`（推荐配置 + 预热 2000 步的余弦衰减、梯度裁剪 1.0、W1 选模）和 `configs/dreamt_large.json`（B1 + 8 层、128 通道）。
- `submit_v2_recommended.sh` 可用 `CONFIG` 环境变量选择配置，输出目录按配置名命名。
- 新增去重 sweep：`configs/sweep_v3.json`（[−1,1] 上 5 个互不重复的 dropout 设置）、`scripts/sweep_config.py`（拒绝等效重复项和非 [−1,1] 的基础配置）、`submit_sweep_v3.sh`（Slurm 数组作业）。

### 测试

新增 `test_distribution_metrics`、`test_training_schedule`、`test_subject_window_evaluation`、`test_sweep_config`，并更新了相关测试。CPU 上 690 个测试全部通过（20 个需要 GPU 的测试跳过）。
