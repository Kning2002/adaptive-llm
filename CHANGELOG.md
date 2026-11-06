# Clean MVP 整理记录

父版本：`code_mvp_20261019`。本次仅创建独立精简版本，不覆盖父版本或原训练工程。

## 命名与组织

- 文件统一 snake_case，修复 `cumstom` 拼写；数据类改为 `PVForecastDataset`。
- 对外模型类为 `AdaptiveLLM`；保留包级 `Model` 别名。
- 合并重复 LLM 包装层为 `FrozenGPT2Encoder`；保留 checkpoint 模块属性名。
- 增加统一 Ruff 格式与静态检查配置，不增加运行依赖。

## 删除与收敛

- 删除未使用的 DS/Prob/Reformer 注意力、Transformer Encoder、额外 embedding、Inception V2 和 ProbMask。
- 删除 AMP guard、旧训练器 failure 装饰器、未使用的 tokenizer/Llama/Reformer 导入。
- 删除全部消融分支，Full 固定周期—残差、LSTM、Prompt V2、GPT-2、TimesBlock 和可学习门。
- 删除未消费的预测输入和数值 decoder 构造，Dataset 返回四个有效字段。
- 同一数据文件不再作为 source/target 域重复读取；每分区读取一次完整 CSV。
- 日历特征只保留当前分钟级五特征，顺序和公式保持不变。
- 指标统一 compute_metrics，删除重复计算与未消费指标。
- 参数全扫描由每个训练 step 移至初始化和 epoch 边界；输出、损失、梯度仍每步检查。
- 保留分区、填补、窗口、批量与评分规则；未增加真实训练或重新评分。

## 修正与兼容

- 非法维度、注意力头数、非有限配置、dropout 和门值提前报错。
- 常量真实值的极差归一化明确报错，不再将不可定义指标写为零。
- CSV 编码 fallback 仅捕获 UnicodeDecodeError，不吞掉其他读取错误。
- 补齐绝对路径扫描测试、周期重构测试及旧版本同权重回归。
- 移除无效旧注意力初始化会改变全新训练的随机数序列；不承诺同 seed 初始化与旧版相同。
- 同权重输出/梯度与数据数值采用明确容差；指标回归要求数值一致。详细结果见 verification.json。
