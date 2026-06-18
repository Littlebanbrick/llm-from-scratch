// English Report Template

// ===== Code Block Style Optimization =====

// 控制代码块外观（背景、边距）
// 只处理"代码块"
#show raw.where(block: true): set block(
  fill: luma(250),
  inset: 6pt,
  radius: 3pt,
)

#show raw.where(block: true): set text(
  size: 9pt,
  font: ("Liberation Mono", "Noto Serif CJK SC"),
)

#show raw.where(block: true): set par(
  leading: 1.15em,
)

// 内联代码：等宽 + 衬线汉字
#show raw.where(block: false): set text(
  font: ("Liberation Mono", "Noto Serif CJK SC"),
)

#set page(
  paper: "a4",
  margin: (left: 2.6cm, right: 2.6cm, top: 2.4cm, bottom: 2.8cm),
  numbering: "1",
  number-align: bottom + center,
  header: context [
    #text(size: 12pt, fill: gray.darken(25%))[deep-learning-from-scratch]
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

// Figure caption: smaller, italic, muted gray — distinct from body text
#show figure.caption: set text(
  size: 11pt,
  font: ("Liberation Serif", "Noto Serif CJK SC"),
  style: "italic",
  fill: luma(90),
)

#show figure.caption: set par(
  leading: 1.0em,
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
  #text(size: 35pt, weight: "bold")[
    Lab 2
    \
  ]
  #v(0%)
  #text(size: 25pt, weight: "bold")[
    NumPy Fundamentals and Manual Neural Network Implementation
  ]
  #image("icon_ZJU.png", width: 40%)
  #v(2em)
  #text(size: 20pt)[Author: Chuanyu Wang]
  #v(0.5em)
  #text(size: 20pt)[Time: 2026-6-16]
  #v(0.5em)
]

#pagebreak()

// Main report starts here

= 零、数学预备知识

本章节统一约定符号系统，并详细推导在 Task 4 中所使用的所有梯度公式。理解本章内容是正确实现反向传播的前提。

== 符号约定

本章使用以下统一符号体系，其定义贯穿全文。对于一个包含 `m` 个样本的 Mini-batch，我们约定：

#figure(
  table(
    columns: 3,
    stroke: 0.3pt,
    align: (left, center, left),
    [变量名], [形状], [含义],
    [X], [(m, 2)], [输入特征矩阵，每行一个样本],
    [y], [(m, 1)], [真实标签 (0 或 1)],
    [W1], [(2, h)], [输入层到隐藏层的权重矩阵],
    [b1], [(1, h)], [隐藏层偏置行向量],
    [W2], [(h, 1)], [隐藏层到输出层的权重矩阵],
    [b2], [(1, 1)], [输出层偏置标量],
    [Z1], [(m, h)], [隐藏层的线性输出（未激活）],
    [A1], [(m, h)], [隐藏层的激活输出（ReLU 后）],
    [Z2], [(m, 1)], [输出层的线性输出（未激活）],
    [A2], [(m, 1)], [输出层的激活输出（Sigmoid 后，即预测概率）],
  ),
  caption: [Task 4 神经网络中所有变量的定义与形状。其中输入为 2 维，输出为 1 维，`m` 为当前 Mini-batch 大小，`h` 为隐藏层神经元数。]
)

此外，我们定义：$ "g1"(z) = max(0, z) $ 为 ReLU 激活函数、$ "g2"(z) = 1 / (1 + e^(-z)) $ 为 Sigmoid 激活函数。

损失函数采用二分类交叉熵： $ L = - 1/m sum_i [ y_i log("A2"_i) + (1-y_i) log(1-"A2"_i) ] $

#pagebreak()

== 前向传播的数学表达

前向传播将输入 `X` 逐步映射为预测概率 `A2`，其计算图可表示为以下依次执行的变换：

1.  `Z1 = X @ W1 + b1`
2.  `A1 = ReLU(Z1)`
3.  `Z2 = A1 @ W2 + b2`
4.  `A2 = Sigmoid(Z2)`

其中 `@` 表示矩阵乘法。偏置向量在加法时通过广播机制扩展至与 `X @ W1` 相同的形状。

== 反向传播的核心思想

反向传播的目标是计算损失函数 `L` 对每个可训练参数 `(W1, b1, W2, b2)` 的偏导数，即梯度。根据微积分中的链式法则，梯度从输出端 `A2` 开始，沿计算图逆向逐层传递。整个推导遵循两条原则：

- 线性层的反向传播遵循 "转置矩阵乘以上游梯度" 的规则（见下文定理）。
- ReLU 激活函数的反向传播需要在正输入处通过梯度，在负输入处截断梯度。

=== 矩阵微积分中的关键定理

对于线性变换 `Z = X @ W`，其中 `X` 形状为 `(m, k)`，`W` 形状为 `(k, n)`，`Z` 形状为 `(m, n)`，若上游梯度为 `dZ`（损失对 `Z` 的偏导），则有以下两个常用结论：

#figure(
  table(
    columns: 2,
    stroke: 0.3pt,
    align: (left, left),
    [目标变量], [公式],
    [损失对 `W` 的梯度], [`dW = X.T @ dZ`],
    [误差向上一层的反向传播], [`dX = dZ @ W.T`],
  ),
  caption: [线性层反向传播的通用公式。其核心逻辑是通过矩阵转置实现误差信号的逆向流动。]
)

该定理的证明可通过对标量链式法则进行向量化展开获得，此处不再展开，但需特别注意矩阵相乘的顺序与转置。

#pagebreak()

== 梯度公式的逐层推导

以下推导使用链式法则，从输出层向输入层逐层展开。设当前 Mini-batch 大小为 $m$，所有除法均为逐元素操作。

=== 输出层 $("Z2" -> "W2", "b2")$

1. 起点：$"dZ2" = (partial L) / (partial"Z2")$，即损失对输出层线性输入的敏感度。其推导过程如下。

   对于单个样本（省略下标 $i$），记 $"A2" = sigma("Z2") = 1 / (1 + e^(-"Z2"))$，单样本交叉熵损失为：
   $L_"sample" = -[y log("A2") + (1-y) log(1-"A2")]$

   由链式法则：
   $(partial L_"sample") / (partial "Z2") = (partial L_"sample") / (partial "A2") * (partial "A2") / (partial "Z2")$

   分别计算两个因子：
   - 交叉熵对 $"A2"$ 的偏导：$(partial L_"sample") / (partial "A2") = (1-y)/(1-"A2") - y/"A2"$
   - Sigmoid 的导数（$sigma'(z) = sigma(z)(1-sigma(z))$）：$(partial "A2") / (partial "Z2") = "A2" * (1 - "A2")$

   代入链式法则并化简：
   $
   (partial L_"sample") / (partial "Z2") \
   = [(1-y)/(1-"A2") - y/"A2"] * "A2" * (1 - "A2") \
   = (1-y)"A2" - y(1-"A2") \
   = "A2" - y
   $

   该结果对每个样本独立成立，推广至整个 Mini-batch 即得矩阵形式：
   $"dZ2" = "A2" - y$ \
   （此处 $"dZ2"$ 不含损失的平均因子 $1/m$，该因子在后续计算 $"dW2", "db2"$ 时显式加入。）

2. 权重梯度：应用线性层定理，输入为 $"A1"$，上游梯度为 $"dZ2"$，故
   $"dW2" = ("A1".T @ "dZ2") / m$。

3. 偏置梯度：$"db2" = 1/m sum_i "dZ2"_i$，即对 Batch 维度求均值。

=== 误差回传至隐藏层 $("Z2" -> "A1")$

应用线性层定理的反向传播部分：
$"dA1" = "dZ2" @ "W2".T$。
此时 $"dA1"$ 形状为 $(m, h)$，与 $"A1"$ 形状一致，表示损失对隐藏层输出的敏感度。

=== ReLU 激活层 $("A1" -> "Z1")$

ReLU 函数 $"A1" = max(0, "Z1")$ 的导数为：
$ (partial "A1"_i) / (partial "Z1"_i) = 1$ 若 $"Z1"_i > 0$，否则 $0$。
因此 $"dZ1" = "dA1" ⊙ ("Z1" > 0)$，其中 $⊙$ 表示逐元素乘法，$("Z1" > 0)$ 为布尔掩码。

=== 输入层 $("Z1" -> "W1", "b1")$

1. 权重梯度：应用线性层定理，输入为 $X$，上游梯度为 $"dZ1"$，故
   $"dW1" = (X.T @ "dZ1") / m$。
2. 偏置梯度：$"db1" = 1/m sum_i "dZ1"_i$，沿 Batch 维度求均值。

== 梯度公式汇总表

为便于查阅，将上述所有梯度公式汇总如下：

#figure(
  table(
    columns: 3,
    stroke: 0.3pt,
    align: (left, center, center),
    [变量], [公式], [形状校验],
    [$"dZ2"$], [$"A2" - y$], [$(m, 1)$],
    [$"dW2"$], [$("A1".T @ "dZ2") / m$], [$(h, 1)$],
    [$"db2"$], [$1/m sum "dZ2"$], [$(1, 1)$],
    [$"dA1"$], [$"dZ2" @ "W2".T$], [$(m, h)$],
    [$"dZ1"$], [$"dA1" * ("Z1" > 0)$], [$(m, h)$],
    [$"dW1"$], [$(X.T @ "dZ1") / m$], [$(2, h)$],
    [$"db1"$], [$1/m sum "dZ1"$], [$(1, h)$],
  ),
  caption: [Task 4 反向传播完整梯度公式表。使用前需确认所有矩阵形状与符号约定一致。]
)

== 关于维度判断的工程准则

在手动实现反向传播时，以下准则可有效避免维度错误：

- 黄金法则：梯度的形状必须严格等于对应参数矩阵的形状。例如 `dW1` 必须与 `W1` 同为 `(2, h)`。
- 矩阵乘法的合法性：`A @ B` 要求 `A` 的列数等于 `B` 的行数。若形状不匹配，绝大多数情况下通过转置即可修正。
- 平均操作的轴选择：偏置梯度 `db` 必须与 `b` 同形，因此需对 Batch 维度（`axis=0`）求平均，并保持维度 `keepdims=True`。

本章所阐述的数学原理为 Task 4 中手动反向传播的完整依据，也是后续使用 PyTorch 验证梯度正确性的理论基准。

#pagebreak()

= 一、Task 1: NumPy 数组操作与向量化

Task 1 旨在熟悉 NumPy 的核心数据结构 `ndarray`，掌握向量化编程的基本范式。通过一系列渐进式操作，我们从随机数组的创建出发，逐步深入到标准化、矩阵乘法、数值稳定激活函数和损失函数计算——这些都是在后续任务中反复出现的基本操作。

== 数组创建与基本统计

首先生成一个形状为 $(100, 5)$ 的随机矩阵 $X$，其元素服从标准正态分布 $cal(N)(0, 1)$：

#```python
X = np.random.randn(100, 5)
```

随后分别计算每列（`axis=0`）和每行（`axis=1`）的均值与标准差：

#```python
col_mean = X.mean(axis=0, keepdims=True)   # shape (1, 5)
col_std  = X.std(axis=0, keepdims=True)    # shape (1, 5)
row_mean = X.mean(axis=1, keepdims=True)   # shape (100, 1)
row_std  = X.std(axis=1, keepdims=True)    # shape (100, 1)
```

使用 `keepdims=True` 保留了原始维度信息，这在后续的广播操作中至关重要。

== 标准化（Standardization）

对矩阵 $X$ 进行列标准化，使得每列均值为 0、标准差为 1：

#```python
X_norm = (X - col_mean) / col_std
```

验证标准化结果：

- 各列均值近似为 $0$（数量级 $10^(-17)$，为浮点精度误差）
- 各列标准差均为 $1$

这证明标准化操作完全正确。标准化是深度学习中最常见的预处理步骤之一，它能确保不同尺度的特征对梯度更新的贡献一致，从而加速收敛。

== 矩阵乘法形状分析

计算 $X @ X.T$ 和 $X.T @ X$，并分析其形状：

- $X$ 的形状为 $(100, 5)$，故 $X @ X.T$ 的形状为 $(100, 100)$，表示样本间的相似度矩阵。
- $X.T @ X$ 的形状为 $(5, 5)$，表示特征间的协方差矩阵（未中心化版本）。

== 数值稳定的 Sigmoid 实现

Sigmoid 函数定义为 $sigma(z) = 1 / (1 + e^(-z))$。直接计算时，当 $z$ 很大或很小时 $e^(-z)$ 或 $e^(z)$ 会溢出。我们采用分段计算策略：

#```python
def sigmoid(z):
    z = np.asarray(z)
    result = np.empty_like(z, dtype=np.float64)
    positive_mask = z >= 0
    # z >= 0: 直接计算 1/(1 + exp(-z))
    result[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))
    # z < 0: 等价形式 exp(z) / (1 + exp(z))
    result[~positive_mask] = np.exp(z[~positive_mask]) / (1.0 + np.exp(z[~positive_mask]))
    return result
```

该实现利用了对负值区域的等价变形：

当 $z < 0$ 时：$1 / (1 + e^(-z)) = e^(z) / (1 + e^(z))$

测试结果验证了其正确性和数值稳定性：

```text
sigmoid([-1000, 0, 1000]) = [0.0, 0.5, 1.0]
```

== 二分类交叉熵损失

生成二进制标签向量 $y$，并使用标准化后的第一列作为预测概率 $p$，计算交叉熵损失：

#```python
eps = 1e-15
p_clipped = np.clip(p, eps, 1 - eps)
loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
```

使用 `np.clip` 将预测概率限制在 $[epsilon, 1-epsilon]$ 内，避免 $log(0)$。最终损失值为 $0.8349$。

#pagebreak()

= 二、Task 2: 线性回归与梯度下降

Task 2 从零开始实现了线性回归的 Mini-batch 梯度下降。这是理解梯度下降算法工作原理的最小可运行实验。

== 合成数据生成

生成 200 个样本，特征 $x$ 在 $[-3, 3]$ 区间均匀分布，目标值 $y$ 遵循线性关系叠加高斯噪声：

#```python
true_w, true_b = 2.0, 1.0
noise = np.random.randn(m, 1) * 0.5
y = true_w * X + true_b + noise
```

数据以 80/20 比例划分为训练集（160 样本）和验证集（40 样本）。

== 模型与梯度推导

线性回归模型：$y_"pred" = w x + b$

损失函数为均方误差（MSE）：$L = 1/m sum_i (y_"pred"_i - y_i)^2$

对参数求偏导：
- $(partial L) / (partial w) = 2/m sum_i x_i (y_"pred"_i - y_i)$
- $(partial L) / (partial b) = 2/m sum_i (y_"pred"_i - y_i)$

在向量化实现中：

#```python
error = y_pred - y_batch
dw = (2 / batch_size) * (X_batch.T @ error)
db = (2 / batch_size) * np.sum(error)
```

== 训练结果

使用学习率 $lr = 0.01$、Batch 大小 32 训练 1000 个 Epoch。

```text
Epoch    0 | Train Loss: 7.5942 | Val Loss: 6.2821
Epoch  100 | Train Loss: 0.2319 | Val Loss: 0.2398
...
Epoch  900 | Train Loss: 0.2319 | Val Loss: 0.2399
```

#figure(
  image("task2_grad_desc.png", width: 80%),
  caption: [Task 2 训练与验证 MSE 损失收敛曲线。两条曲线均在约 100 个 Epoch 后趋于稳定，且几乎重合，说明模型既没有欠拟合也没有过拟合。]
)

最终学习到的参数与真实值的对比如下：

#figure(
  table(
    columns: 3,
    stroke: 0.3pt,
    align: (center, center, center),
    [参数], [学习值], [真实值],
    [$w$], [1.9909], [2.0],
    [$b$], [1.0311], [1.0],
  ),
  caption: [线性回归参数对比。学习值在真实值的邻域内，偏差来自采样噪声和随机梯度下降的随机性。]
)

== 不可约误差：损失为何不降为 0

最终训练损失和验证损失均稳定在 $approx 0.23$，而非 0。这是由数据生成过程中的噪声决定的。噪声标准差为 0.5，方差为 $0.5^2 = 0.25$。即使模型完美拟合了 $y = 2x + 1$，损失也等于噪声方差：

$L_"min" = 1/m sum_i "noise"_i^2 approx "Var"("noise") = 0.25$

实验得到的 $0.2319$ 略低于理论下限 $0.25$，是因为 200 个采样点的平均平方噪声恰好偏小，属于采样巧合。这一现象揭示了一个重要的工程认知：损失降不下去，不一定是代码写错了，可能是数据本身的不可约噪声在起作用。强行增加轮数或模型复杂度只会导致过拟合，而非降低损失。

#pagebreak()

= 三、Task 3: 逻辑回归与二分类

Task 3 将任务从回归扩展到二分类。使用非线性可分的数据集和逻辑回归模型，验证线性决策边界在复杂分类任务上的局限性。

== 数据集

使用 `sklearn.datasets.make_moons` 生成 300 个样本的"月牙"形数据，噪声级别 0.2。该数据集的两个类别呈交错月牙形态，并非线性可分。

#```python
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
```

数据以 80/20 划分训练集和测试集，并进行标准化预处理。

== 模型实现

逻辑回归模型：$p = sigma(X @ w + b)$，采用二分类交叉熵损失。

梯度公式如下：

#```python
dz = p - y_batch                    # 输出层误差
dw = (X_batch.T @ dz) / batch_size  # 权重梯度
db = np.mean(dz, axis=0, keepdims=True)  # 偏置梯度
```

训练超参数：学习率 0.1、Batch 大小 32、Epoch 数 2000。

== 训练结果

```text
Epoch    0 | Train Loss: 0.5698 | Test Loss: 0.5506 | Train Acc: 0.8417 | Test Acc: 0.9167
Epoch 1000 | Train Loss: 0.2944 | Test Loss: 0.2537 | Train Acc: 0.8792 | Test Acc: 0.9000
Epoch 1999 | Train Loss: 0.2944 | Test Loss: 0.2540 | Train Acc: 0.8792 | Test Acc: 0.9000
```

#figure(
  image("task3_accuracy_and_loss.png", width: 80%),
  caption: [Task 3 训练与测试损失及准确率曲线。训练损失在约 100 个 Epoch 后即收敛，准确率稳定在 87.92% / 90.00%。]
)

== 决策边界分析

#figure(
  image("task3_decision_boundary.png", width: 80%),
  caption: [Task 3 逻辑回归决策边界。黑色等高线为 p = 0.5 的决策面。由于逻辑回归是线性模型，决策边界呈一条直线，无法完美分离交错月牙形数据。]
)

测试准确率 90.00% 是线性模型在该数据集上的上限。非线性结构（如两个月牙的交叉区域）超出了线性模型的表达范围。这是深度学习引入非线性激活函数的核心动机——没有隐藏层的网络只能学到直线，而真实世界的决策边界几乎从来不是直线。

#pagebreak()

= 四、Task 4: 单隐藏层神经网络

Task 4 构建了一个包含单个隐藏层的全连接神经网络，从零实现前向传播、手动反向传播和 Mini-batch 梯度下降，并结合 PyTorch 自动微分验证梯度正确性。

== 网络架构

网络的层间结构为 $2 -> 10 -> 1$:

#figure(
  table(
    columns: 3,
    stroke: 0.3pt,
    align: (left, center, center),
    [层], [形状变换], [激活函数],
    [输入], [`(m, 2)`], [—],
    [隐藏层], [`(m, 10)`], [ReLU],
    [输出层], [`(m, 1)`], [Sigmoid],
  ),
  caption: [Task 4 神经网络各层结构。隐藏层包含 10 个神经元，输出层通过 Sigmoid 将线性输出压缩至 (0,1) 区间作为概率。]
)

激活函数的选取并非偶然。输出层使用 Sigmoid 是因为二分类的最终输出必须是概率，需要将实数压缩到 $(0,1)$ 区间；隐藏层使用 ReLU 则是出于训练效率——Sigmoid 的导数在两端趋于 $0$，经多层链式传播后极易造成梯度消失，而 ReLU 在正区间的导数为常数 $1$，梯度可以无损传递。同时 $max(0, z)$ 的计算仅需一次比较，远快于 Sigmoid 的指数运算。这一"输出层 Sigmoid、隐藏层 ReLU"的搭配已成为现代神经网络的事实标准。

训练使用 500 个 `make_moons` 样本（噪声 0.2），以 80/20 划分训练与测试集。超参数：学习率 0.01、Batch 大小 32、Epoch 数 3000。

权重初始化采用 He 初始化（针对 ReLU 设计）：

#```python
W1 = np.random.randn(2, 10) * np.sqrt(2 / 2)    # sqrt(2 / n_in)
W2 = np.random.randn(10, 1) * np.sqrt(2 / 10)   # sqrt(2 / n_hidden)
```

== 手动反向传播

前向传播的四步计算如下：

#```python
Z1 = X_batch @ W1 + b1          # (batch_m, 10)
A1 = np.maximum(0, Z1)          # (batch_m, 10), ReLU
Z2 = A1 @ W2 + b2               # (batch_m, 1)
A2 = sigmoid(Z2)                 # (batch_m, 1)
```

反向传播的梯度完全对应于"零、数学预备知识"中的逐层推导：

#```python
# 输出层梯度
dZ2 = A2 - y_batch
dW2 = (A1.T @ dZ2) / batch_m
db2 = np.mean(dZ2, axis=0, keepdims=True)

# 反向传播至隐藏层
dA1 = dZ2 @ W2.T

# ReLU 门控
dZ1 = dA1 * (Z1 > 0)

# 输入层梯度
dW1 = (X_batch.T @ dZ1) / batch_m
db1 = np.mean(dZ1, axis=0, keepdims=True)
```

注意 $"dZ1" = "dA1" * ("Z1" > 0)$ 中的布尔掩码 $("Z1" > 0)$——ReLU 在正区间保持梯度不变，在负区间永久截断梯度。这一简单的"门控"机制是缓解深度网络梯度消失问题的关键设计。

== 训练结果与决策边界

```text
Epoch    0 | Train Loss: 1.2915 | Test Loss: 1.3310 | Train Acc: 0.2225 | Test Acc: 0.1700
Epoch 1500 | Train Loss: 0.1271 | Test Loss: 0.1496 | Train Acc: 0.9550 | Test Acc: 0.9500
Epoch 2700 | Train Loss: 0.0733 | Test Loss: 0.0827 | Train Acc: 0.9825 | Test Acc: 0.9900
```

#figure(
  image("task4_loss_and_accu_curves.png", width: 80%),
  caption: [Task 4 训练与测试损失及准确率曲线。损失持续下降、准确率稳步提升。最终测试准确率达 99%。]
)

#figure(
  image("task4_decision_boundary.png", width: 80%),
  caption: [Task 4 神经网络决策边界。与 Task 3 的直线边界不同，隐藏层赋予模型拟合非线性决策面的能力，黑色 p = 0.5 等高线完美贴合月牙形数据分布。]
)

与 Task 3（逻辑回归）的对比清晰地揭示了隐藏层的价值：

- *Task 3*（无隐藏层）：测试准确率 *90.00%*，决策边界为直线，无法处理月牙形交错。
- *Task 4*（10 个隐藏神经元）：测试准确率 *99.00%*，决策边界为非线性曲线，几乎完美分离两个类别。

从 $90%$ 到 $99%$ 的跨越，正是激活函数 $"ReLU"$ 赋予模型非线性表达能力的结果。

== PyTorch 梯度验证

完成手动训练后，我们在 PyTorch 中重建完全相同的网络结构，将训练好的权重同步至 PyTorch 模型，对同一 Batch 进行一次前向和反向传播，逐层比较梯度。

#```python
pt_model = PTModel()
# 注意：PyTorch 的 Linear 层权重形状为 (out_features, in_features)
# 而我们的 W1 形状为 (2, 10)，因此需要转置
pt_model.fc1.weight.data = torch.tensor(W1.T, dtype=torch.float32)
pt_model.fc2.weight.data = torch.tensor(W2.T, dtype=torch.float32)
# ...
pt_loss = loss_fn(pt_pred, batch_y)
pt_loss.backward()
```

#figure(
  table(
    columns: 3,
    stroke: 0.3pt,
    align: (center, center, center),
    [梯度变量], [手动实现], [PyTorch],
    [$"dW1"$], [$(2 times 10)$], [$(2 times 10)$],
    [$"db1"$], [$(1 times 10)$], [$(1 times 10)$],
    [$"dW2"$], [$(10 times 1)$], [$(10 times 1)$],
    [$"db2"$], [$(1 times 1)$], [$(1 times 1)$],
  ),
  caption: [手动实现与 PyTorch 自动微分的梯度形状完全一致。]
)

#figure(
  table(
    columns: 2,
    stroke: 0.3pt,
    align: (left, center),
    [梯度变量], [最大绝对差值],
    [$"dW1"$], [$1.59 times 10^(-8)$],
    [$"db1"$], [$2.18 times 10^(-8)$],
    [$"dW2"$], [$1.10 times 10^(-8)$],
    [$"db2"$], [$5.71 times 10^(-9)$],
  ),
  caption: [手动实现与 PyTorch 自动微分的梯度差值对比。所有参数的最大绝对差值均在 $10^(-8)$ 量级，远小于验收阈值 $10^(-6)$，验证通过。]
)

四个梯度的最大绝对差值均在 $10^(-8) 至 10^(-9)$ 量级，远小于实验要求的小于 $10^(-6)$ 的验收阈值。这一结果有力地证明了我们对链式法则的推导和代码实现是完全正确的。微小差异来自浮点运算顺序不同和 PyTorch 内部计算图的实现细节。

#pagebreak()

= 五、总结

本实验从 NumPy 的基础操作出发，逐步构建了线性回归、逻辑回归和单隐藏层神经网络，涵盖了深度学习中三个核心主题：

*向量化编程*（Task 1）：通过 NumPy 的向量化操作替代显式循环，实现了高效的数据预处理和损失计算，培养了"矩阵思维"。

*梯度下降算法*（Task 2）：从零实现了 Mini-batch SGD，直观观察到参数收敛到真实值附近的过程。实验中验证了不可约误差的存在——损失的下限由数据噪声的方差决定，而非模型的能力。

*线性模型的局限*（Task 3）：逻辑回归在 make_moons 上仅能达到 90% 的测试准确率。线性决策边界无法区分交错月牙形的两个类别，这为引入非线性激活函数提供了最直接的动机。

*深度学习的核心机制*（Task 4）：通过在逻辑回归之上添加一层 ReLU 隐藏层，准确率从 90% 跃升至 99%，决策边界从直线扭曲为贴合数据分布的曲线。反向传播的每一步都经 PyTorch 自动微分严格验证，梯度差值在 $10^(-8)$ 量级，确认了推导和实现的正确性。

四组实验串成一条清晰的技术路线：向量化操作 $->$ 梯度下降 $->$ 线性分类器 $->$ 非线性分类器。这不仅仅是四个独立的编程练习，更是一次对深度学习工作原理的完整实践——所有看似复杂的高级网络，最终都可以拆解为这些基本组件的有机组合。
