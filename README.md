# AdaptiveLLM — Clean MVP

当前图示 Full 主模型的独立代码包。只保留训练、验证、测试所需代码，不包含对比模型、消融变体、数据集、实验结果或预训练权重。

AI 编码代理开始工作前请阅读 [AGENT.md](AGENT.md)；`AGENTS.md` 为对应的入口文件。

## 目录

```text
configuration.py            唯一配置定义与参数校验
configs/pvod.json           PVOD 输入字段与训练参数示例
run.py                      check / train / test 入口
models/
  adaptive_llm.py            AdaptiveLLM、Prompt V2、冻结 GPT-2 编码器
  components.py              周期分解、TimesBlock、协变量分支
layers/
  attention.py              解码器使用的注意力
  decoder.py                Transformer Decoder
  embedding.py              数值、位置与日历嵌入
  conv_blocks.py            TimesBlock 使用的卷积模块
data_provider/
  dataset.py                PVForecastDataset 与 CSV 读取
  data_factory.py           DataLoader 批量策略
utils/
  metrics.py                统一指标入口 compute_metrics
  numerical_safety.py       有限值检查
  time_features.py          五个连续日历特征
  masking.py                因果注意力掩码
tests/
  test_mvp.py               离线功能测试
  test_legacy_parity.py     可选的旧 MVP 同权重回归
pyproject.toml              统一格式与静态检查规则
verification.json           本次验证结果
manifest.json               文件 SHA-256 与大小清单
```

## 安装

Python 3.10+。按本机 CPU/CUDA 环境准备 PyTorch，然后安装运行依赖：

```bash
python -m pip install -r requirements.txt
```

运行依赖仅五项：PyTorch、Transformers、NumPy、pandas、scikit-learn。实际验证环境见 `verification.json`，版本范围未穷举组合测试。

准备本地 GPT-2 目录，包含 `config.json` 和匹配权重。默认 6 层、768 维；不自动下载，不需要 tokenizer 文件。

## 数据与配置

CSV 包含 `date` 或 `timestamp`，以及配置中的数值字段。`target` 默认为 `Active_Power`；输入时会将目标列排列到最后。字段列表在 JSON 的 `feature_cols` 中修改。

PVOD 示例采用六个实测辅助变量与功率目标，要求预先将原始 MW 功率转换为 kW，并准备 UTC+8 时间戳。代码不猜测或自动换算单位、时区。

默认参数：15 min 数据、96→96、50%/25%/25%、batch 16、seed 4213、最多 20 epochs、patience 3、FP32、Adam、Huber 与 Prompt 多样性正则。完整可配置项见 `configuration.py`。

只支持 Full：目标归一化 → 周期—残差分解 → 周期 LSTM / 残差重编程与 Prompt V2 → 冻结 GPT-2 → 门控融合 → 协变量 TimesBlock 与预测解码器 → 反归一化。没有消融开关、AMP 路径或其他骨干选择。

## 检查、训练与测试

将下列占位路径换为实际路径；输出目录必须不存在：

```bash
python run.py --config configs/pvod.json --data /path/to/station.csv --llm-path /path/to/gpt2 --output /path/to/check_run --mode check

python run.py --config configs/pvod.json --data /path/to/station.csv --llm-path /path/to/gpt2 --output /path/to/train_run

python run.py --config configs/pvod.json --data /path/to/station.csv --llm-path /path/to/gpt2 --output /path/to/test_run --mode test --checkpoint /path/to/train_run/best.pt
```

支持 `--device cpu` 或 `--device cuda:0`，默认自动选择。训练使用验证预测损失选择 `best.pt`，结束后载入最优权重测试；每轮学习率按既有 PEMS 规则衰减。每步检查预测、损失和梯度；完整参数与 buffer 检查放在训练前及 epoch 边界。

## 保留的数据与评分协议

- CSV 在切分前 ffill，再 bfill；全局 StandardScaler 仅拟合训练段。模型内部窗口归一化另行保留。
- 验证和测试窗口可读取前一分区末尾历史；标签留在当前分区。
- 按保留行构建窗口，不新增重采样、日期删除或时间连续性过滤。输入应事先整理为 15 min 序列。
- 训练和验证保持 shuffle、drop_last；测试按顺序、batch 1。
- 主评分每隔 `pred_len` 个测试窗口起点取一个完整窗口；全部窗口指标另列。
- nMAE = MAE / 真实值极差；NMSE = MSE / 极差²；nRMSE = √NMSE。真实值极差接近零时明确报错，不返回零误差。
- 保留夜间零值与负预测，不裁剪或过滤。

每个分区只读取一次完整 CSV；移除了未消费的额外预测输入。模型的外部 `forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)` 仍保留，`x_dec` 与 `mask` 不参与 Full 计算。

## 输出

`config.json`、`runtime.json` 保存生效参数、设备、数据与 GPT-2 配置指纹；`history.json` 与 `best.pt` 保存损失和验证最优权重。`prediction.npy`、`truth.npy` 保存全部窗口，形状为 `[N,pred_len,1]`；`primary_window_indices.npy`、`metrics.json` 分别记录主评分索引和两套指标；成功结束时生成 `COMPLETE.json`。

## 测试与代码风格

普通离线测试在临时目录生成小型随机 GPT-2 和合成 CSV：

```bash
python -B -m unittest discover -s tests -v
```

默认 8 项功能测试运行，1 项旧版本对照测试在未指定来源时跳过。要执行同权重回归，在 PowerShell 中指定未经修改的旧 MVP：

```powershell
$env:ADAPTIVELLM_LEGACY_ROOT = (Resolve-Path ../code_mvp_20261019).Path
python -B -m unittest discover -s tests -v
```

回归覆盖 state_dict 严格加载、前向输出、Prompt 正则、梯度、三套分区/标签长度配置、窗口边界、标准化与反标准化、指标一致性。数值容差记录在测试及 `verification.json` 中，不要求跨进程浮点逐位相等。

开发检查：

```bash
python -m pip install -r requirements-dev.txt
ruff check .
ruff format --check .
```

文件和函数统一 snake_case，类使用 PascalCase。保留 `LLM_encoder`、`tokenConv` 等历史模块属性名以兼容权重键；`from models import Model` 作为 `AdaptiveLLM` 的别名继续可用。

移除曾被创建后立即覆盖的旧注意力层，改变了随机数消耗顺序，因此相同 seed 的全新训练初始化不再与旧版逐位一致；已有 Full 权重的键名和计算结构保持兼容。本轮没有启动真实数据训练。

## 来源与许可

本包来自上一版 MVP 的独立精简副本。原工程、旧代码包和论文结果均未修改。发布前请按上游项目许可补充适用的 LICENSE 与署名；本包不推定上游授权类型。
