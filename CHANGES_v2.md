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
