// English Report Template

// ===== Code Block Style Optimization =====

// 控制代码块外观（背景、边距）
// 只处理“代码块”
#show raw.where(block: true): set block(
  fill: luma(250),
  inset: 6pt,
  radius: 3pt,
)

#show raw.where(block: true): set text(
  size: 9pt,
  font: ("Noto Sans Mono CJK SC", "Droid Sans Fallback"),
)

#show raw.where(block: true): set par(
  leading: 1.15em,
)

#set page(
  paper: "a4",
  margin: (left: 2.6cm, right: 2.6cm, top: 2.4cm, bottom: 2.8cm),
  numbering: "1",
  number-align: bottom + center,
  header: context [
    #text(size: 12pt, fill: gray.darken(25%))[llm-from-scratch]
    #h(1fr)
    #text(size: 12pt, fill: gray.darken(25%))[#datetime.today().display("[month repr:short] [day], [year]")]
  ],
  footer: context align(center)[
    #text(size: 11pt, fill: gray.darken(50%))[#counter(page).display()]
  ],
)

// Typography: classic academic style
#set text(
  font: ("Liberation Serif", "Noto Serif CJK SC"),
  size: 14pt,
  lang: "en",
)

// Paragraph style for English reports
#set par(
  justify: true,
  first-line-indent: 0em,
  leading: 0.75em,
  spacing: 1em,
)

// Heading hierarchy
#set heading(numbering: "1.")
#show heading.where(level: 1): it => [
  #v(0.9em)
  #text(size: 20pt, weight: "bold", it.body)
  #v(0.35em)
]
#show heading.where(level: 2): it => [
  #v(0.55em)
  #text(size: 15pt, weight: "semibold", it.body)
  #v(0.25em)
]
#show heading.where(level: 3): it => [
  #v(0.35em)
  #text(size: 13pt, weight: "semibold", it.body)
  #v(0.15em)
]

// Cover page
#align(center + horizon)[
  #v(0%)
  #text(size: 35pt,weight: "bold")[
    Lab 1
    \
  ]
  #v(0%)
  #text(size: 25pt,weight: "bold")[
    MNIST Handwritten Digit Recognition
    \
    and Hyperparameter Exploration
  ]
  #image("icon_ZJU.png", width: 40%)
  #v(2em)
  #text(size: 20pt)[Author: Chuanyu Wang]
  #v(0.5em)
  #text(size: 20pt)[Time: 2026-6-14]
  #v(0.5em)
]

#pagebreak()

// Main report starts here

= 零、硬件配置和软件环境

== 硬件配置

- CPU: Intel(R) Core(TM) Ultra 9 285H (4 cores)
- 内存: 8 GB DDR4

== 软件环境

- 操作系统: Ubuntu 22.04 LTS
- Python 版本: 3.10.20
- 主要依赖库及版本:

```
PyTorch version: 2.12.0+cpu
TorchVision version: 0.27.0+cpu
Matplotlib version: 3.10.9
NumPy version: 2.2.5
scikit-learn version: 1.7.2
```

== 说明

- 所有实验均在 CPU 上运行，未使用 GPU 加速。
- 随机种子固定为 `torch.manual_seed(42)` 以保证可重复性。

#pagebreak()

= 一、基准测试

== 基准模型配置

#figure(
  table(
    columns: 2,
    stroke: 0.3pt,
    align: (left, left),
    [配置项], [值 / 说明],
    [输入层], [784 (28×28 像素展平)],
    [隐藏层], [2 层，每层 128 个神经元，激活函数 ReLU],
    [输出层], [10 个神经元 (对应数字 0~9)],
    [`loss func`], [交叉熵损失 (`CrossEntropyLoss`)],
    [优化器], [Adam],
    [`lr`], [0.001],
    [`batch_size`], [64],
    [`epochs`], [20],
    [正则化], [无 Dropout、无 Batch Normalization、无权重衰减],
    [数据划分], [训练集 54,000 张，验证集 6,000 张 (从原始训练集分出 10%)，测试集 10,000 张],
  ),
  caption: [基准模型超参数配置]
)

== 训练过程记录

以下为基准模型在每个 epoch 结束后的训练/验证损失与准确率：

```
Epoch  1/20 | Train Loss: 0.3368 | Train Acc: 0.9032 | Val Loss: 0.1884 | Val Acc: 0.9452
Epoch  2/20 | Train Loss: 0.1347 | Train Acc: 0.9594 | Val Loss: 0.1222 | Val Acc: 0.9638
Epoch  3/20 | Train Loss: 0.0910 | Train Acc: 0.9721 | Val Loss: 0.1069 | Val Acc: 0.9655
Epoch  4/20 | Train Loss: 0.0673 | Train Acc: 0.9794 | Val Loss: 0.0967 | Val Acc: 0.9718
Epoch  5/20 | Train Loss: 0.0527 | Train Acc: 0.9831 | Val Loss: 0.0865 | Val Acc: 0.9753
Epoch  6/20 | Train Loss: 0.0423 | Train Acc: 0.9861 | Val Loss: 0.0877 | Val Acc: 0.9748
Epoch  7/20 | Train Loss: 0.0318 | Train Acc: 0.9895 | Val Loss: 0.1061 | Val Acc: 0.9703
Epoch  8/20 | Train Loss: 0.0274 | Train Acc: 0.9913 | Val Loss: 0.0880 | Val Acc: 0.9770
Epoch  9/20 | Train Loss: 0.0217 | Train Acc: 0.9933 | Val Loss: 0.1056 | Val Acc: 0.9707
Epoch 10/20 | Train Loss: 0.0215 | Train Acc: 0.9928 | Val Loss: 0.0979 | Val Acc: 0.9753
Epoch 11/20 | Train Loss: 0.0159 | Train Acc: 0.9949 | Val Loss: 0.0998 | Val Acc: 0.9768
Epoch 12/20 | Train Loss: 0.0165 | Train Acc: 0.9944 | Val Loss: 0.1127 | Val Acc: 0.9713
Epoch 13/20 | Train Loss: 0.0121 | Train Acc: 0.9958 | Val Loss: 0.1037 | Val Acc: 0.9763
Epoch 14/20 | Train Loss: 0.0112 | Train Acc: 0.9962 | Val Loss: 0.1219 | Val Acc: 0.9763
Epoch 15/20 | Train Loss: 0.0128 | Train Acc: 0.9957 | Val Loss: 0.1176 | Val Acc: 0.9747
Epoch 16/20 | Train Loss: 0.0119 | Train Acc: 0.9959 | Val Loss: 0.1163 | Val Acc: 0.9767
Epoch 17/20 | Train Loss: 0.0106 | Train Acc: 0.9965 | Val Loss: 0.1492 | Val Acc: 0.9683
Epoch 18/20 | Train Loss: 0.0086 | Train Acc: 0.9971 | Val Loss: 0.1255 | Val Acc: 0.9778
Epoch 19/20 | Train Loss: 0.0103 | Train Acc: 0.9966 | Val Loss: 0.1313 | Val Acc: 0.9760
Epoch 20/20 | Train Loss: 0.0097 | Train Acc: 0.9968 | Val Loss: 0.1202 | Val Acc: 0.9782
Test accuracy: 0.9785
```

== 损失曲线与准确率曲线分析

#figure(
  image("original.png", width: 80%),
  caption: [基准模型训练/验证损失曲线]
)

**观察结论**:
- 训练损失快速下降，并在第 10 轮后稳定在 0.02 以下，表明模型在训练集上拟合良好。
- 验证损失在前 10 轮迅速下降并趋于平稳，但在第 15-20 轮之间出现了明显的轻微反弹（从 0.10 升至 0.15 左右）。 虽然整体未出现极度严重的过拟合，但这种趋势通常暗示模型已进入早期过拟合阶段。
- 验证准确率在 20 轮内达到 97.8%，与测试准确率 97.85% 高度一致，表明模型泛化能力较好。
- 验证损失在小范围内锯齿状波动属于正常现象，源于优化过程的非单调性及小批量随机性。

== 基准模型的意义

基准模型 (`hidden_layers=[128,128]`, 无正则化) 作为后续所有超参数对比实验的参考点。其表现证明了：
- 一个简单的两层 MLP 足以在 MNIST 上达到 97% 以上的准确率。
- MNIST 任务对全连接网络而言相对容易，但仍有提升空间（如通过卷积网络可达到 99%+）。

#pagebreak()

= 二、实验A：更改隐藏层数

本实验固定每层神经元数量为128，激活函数为ReLU，不使用任何正则化手段，仅改变隐藏层的数量（0层、1层、2层、3层），观察模型容量对训练过程及泛化能力的影响。其中2层隐藏层的基准模型结果已在前一章给出，以下分别展示0层、1层、3层的训练曲线及分析。

== 0个隐藏层（逻辑回归）

#figure(
  image("no_layer.png", width: 80%),
  caption: [0个隐藏层（逻辑回归）的训练/验证损失与准确率曲线]
)

训练损失从0.56快速下降，最终稳定在0.24左右，验证损失从0.36下降至0.28附近后趋于平坦，未见上升。训练准确率最终达到93.3%，验证准确率约91.7%，测试准确率为92.57%。训练损失与验证损失之间存在约0.04的差距，且准确率差距约1.6%，说明线性模型在MNIST上存在容量不足的问题，无法进一步拟合更复杂的模式。但验证曲线平滑稳定，没有过拟合迹象。

== 1个隐藏层

#figure(
  image("one_layer.png", width: 80%),
  caption: [1个隐藏层（128神经元）的训练/验证损失与准确率曲线]
)

训练损失几乎降至0.0037，训练准确率高达99.94%，模型在训练集上达到近乎完美的拟合。验证损失迅速下降至0.08左右，并长期保持平稳，仅在末尾略有升高至0.094，但仍远低于0层模型的验证损失。验证准确率稳定在97.7%附近，测试准确率为97.64%。相比0层模型，引入一个隐藏层后准确率提升了约5个百分点，表明非线性激活函数带来的表达能力至关重要。验证曲线未见明显上升，泛化能力优良。

== 2个隐藏层（基准）

该模型已在第一章中详细分析。训练损失降至0.0097，准确率99.68%；验证损失在0.12左右波动，验证准确率97.82%，测试准确率97.85%。相比1层模型，2层模型的验证损失略有升高（从0.08升至0.12），但测试准确率几乎持平，说明增加至第二层带来的收益非常有限。

== 3个隐藏层

#figure(
  image("three_layers.png", width: 80%),
  caption: [3个隐藏层的训练/验证损失与准确率曲线]
)

训练损失迅速下降至0.01附近，训练准确率接近100%，模型在训练集上发生高度拟合。验证损失在前10轮快速下降至0.09左右，但第12轮后出现明显的持续上升趋势（从0.09攀升至0.13以上），验证准确率在峰值约98%后于最后几轮下降至约97.7%。这是典型的过拟合信号：模型容量过大，在有限数据上开始记忆噪声，导致泛化能力衰退。

== 综合结论

对于MNIST数据集，从0层到1层隐藏层，模型性能获得巨大提升（92.6% → 97.6%），说明非线性是必要的。从1层到2层，性能提升微弱（97.6% → 97.8%），进一步加深至3层反而因过拟合导致验证损失上升、验证准确率下降。因此，在该任务上1-2层隐藏层已足够，更深的全连接网络需要配合正则化手段（如Dropout、早停）才能发挥潜力，否则只会浪费算力并损害泛化能力。

= 三、