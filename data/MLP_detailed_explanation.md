训练 MLP（以及扩展的神经网络）的权重依赖于四个主要部分。

> 训练神经网络的步骤：
>
> 1. **前向传播**：计算一个输入示例的网络输出。
> 2. **误差计算**：计算网络预测与目标之间的预测误差。
> 3. **反向传播**：根据输入和权重以相反的顺序计算梯度。
> 4. **参数更新**：使用随机梯度下降更新网络的权值，以减少该实例的误差。`

### 前向传播

训练 MLP 的第一步是计算来自数据集示例的网络输出。我们使用以 $\sigma(x)$ 表示的 sigmoid 函数作为 MLP 的激活函数。可以认为它是一个平滑的阶跃函数，如图。此外，它是连续可微的，这是反向传播的一个理想性质。sigmoid 函数的定义为：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

![1](./img/3/1.png)

前向传播过程的目标是计算特定示例 $x$ 的当前网络输出，将每个输出连接为下一层神经元的输入

为了便于标记和计算，该层的权重被组合成单个权重矩阵 $W_i$，表示该层中权值的集合，其中 $i$ 是层数。层计算对各权值的线性变换是 $x$与 $y$ 之间的内积计算。这种类型通常被称为“全连通”“内积”或“线性”层，因为权重将每个输入连接到每个输出。对于其中 $h_1$ 和 $h_2$ 代表各自层输出的示例 $x$，预测 $\hat{y}$ 就变成：

$$
\begin{aligned}
h_1 &= f(W_1 x + \mathbf{b_1}{}) \\
h_2 &= f(W_2 h_1 + b_2) \\
\hat{y} &= h_2
\end{aligned}
$$

注意，偏置项 $\mathbf{b_1}$ 是一个向量，因为每一层神经元都有一个偏置值。在输出层只有一个神经元，所以偏置项 $b_2$ 是一个标量

在前向传播步骤结束时，我们对网络的输出进行了预测。一旦网络被训练，一个新的例子通过前向传播被评估

### 误差计算

误差计算步骤验证了我们的网络在给出的示例上的执行情况。我们使用均方误差（MSE）作为本例中使用的损失函数（将训练视为回归问题）。MSE定义为

$$
E(\hat{y}, y) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

**注意**：1/2项的存在简化了反向传播的计算。

对于一个单一的输出，误差简化为：

$$
E(\hat{y}, y) = \frac{1}{2} (\hat{y} - y)
$$

MSE通常用于回归问题，测量目标的平方误差的平均值。特性如下：

- 平方函数迫使误差为非负的，保证正负方向的偏差都会计算出正确的损失
- 函数为二次损失，对小误差不敏感，对大误差极其敏感；光滑且可导，有唯一的全局最小值
- 误差的惩罚呈非线性增长

下图展示了输出神经元的前向传播步长和误差传播。

![](./img/3/2.png)

### 反向传播

前向传播过程中，通过输入 $x$ 和网络参数 $\theta$ 计算输出预测 $\hat{y}$。为了提高预测精度，我们可以使用随机梯度下降（SGD）来降低整个网络的误差。

通过微积分的链式法则可以确定每个参数的误差。我们可以使用微积分的链式法则，以前向传播的相反顺序来计算每一层（和运算）的导数，如图所示：

![img](./img/3/3.png)

在我们的前一个例子中，预测 $\hat{y}$ 依赖于 $W_2$。我们可以利用链式法则计算出关于 $W_2$ 的预测误差：

$$
\frac{\partial E}{\partial W_2} = \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2}
$$

> 链式法则允许我们计算误差函数对每个可学参数 $\theta$ 的梯度，从而使用随机梯度下降法更新网络：

我们首先计算与预测相关的输出层的梯度：

$$
\nabla_y E(\hat{y}, y) = \frac{\partial E}{\partial \hat{y}} = (\hat{y} - y)
$$

然后我们可以计算关于第二层参数的误差：
我们目前有"激活后"梯度，所以我们需要计算激活前梯度：（使用逐元素乘法计算梯度与激活函数导数的积）

$$
\begin{aligned}
\nabla_{\mathbf{a}_{2}} E=\frac{\partial E}{\partial\mathbf{a}_{2}} &=\frac{\partial E}{\partial\hat{y}}\cdot\frac{\partial\hat{y}}{\partial\mathbf{a}_{2}}\\
& =\frac{\partial E}{\partial\hat{y}}\odot f^{\prime}\left(\mathbf{W}_{2}\mathbf{h}_{1}+\mathbf{b}_{2}\right)
\end{aligned}
$$

then, 计算关于 $\mathbf{W}_2$ 和 $\mathbf{b_2}$ 的误差：

$$
\begin{aligned}
\nabla_{\mathbf{W}_{2}} E = \frac{\partial E}{\partial\mathbf{W}_{2}} &= \frac{\partial E}{\partial\hat{y}}\cdot\frac{\partial\hat{y}}{\partial\mathbf{a}_{2}}\cdot\frac{\partial\mathbf{a}_{2}}{\partial\mathbf{W}_{2}} \\
&= \frac{\partial E}{\partial\mathbf{a}_{2}}\mathbf{h}_{1}^{\top}
\end{aligned}
$$

$$
\begin{aligned}
\nabla_{\mathbf{b}_{2}} E=\frac{\partial E}{\partial\mathbf{b}_{\mathbf{2}}} &=\frac{\partial E}{\partial\hat{y}}\cdot\frac{\partial\hat{y}}{\partial\mathbf{a}_{2}}\cdot\frac{\partial\mathbf{a}_{2}}{\partial\mathbf{b}_{2}}\\
& =\frac{\partial E}{\partial\mathbf{a}_{\mathbf{2}}}
\end{aligned}
$$

我们还可以计算关于第2层的输入（第1层的激活后输出）的误差：

$$
\begin{aligned}
\nabla_{\mathbf{h}_{1}} E=\frac{\partial E}{\partial\mathbf{h}_{\mathbf{1}}} &=\frac{\partial E}{\partial\hat{y}}\cdot\frac{\partial\hat{y}}{\partial\mathbf{a}_{2}}\cdot\frac{\partial\mathbf{a}_{2}}{\partial\mathbf{h}_{1}}\\
& =\mathbf{W}_{2}^{\top}\frac{\partial E}{\partial\mathbf{a}_{2}}
\end{aligned}
$$

然后，我们重复这个过程来计算误差关于第1层的参数 $\mathbf{W}_1$ 和 $b_1$ 的梯度，从而将误差向后传播到整个网络。

![](./img/3/4.png)

### 参数更新

训练过程的最后一步是参数更新。获得关于网络的所有可学习参数的梯度后，我们可以完成单个SGD步骤，根据学习率α更新参数：

$$
\theta = \theta - \alpha \nabla_{\theta} E
$$

这里介绍的SGD更新规则的简单性是有代价的。α的值在SGD中尤为重要，影响：**收敛速度**，**收敛的质量**，甚至**网络收敛的能力**。如果**学习率过小**，网络收敛速度就会非常慢，而且可能会在随机权重初始化附近陷入局部极小值；如果**学习率过大**：权重可能增长更快，变得不稳定，根本无法收敛。

---

# 梯度下降

我们的目标是找到最小化 $E_{\text{train}}$ 的权重 $w$，并且最小化 $E_{\text{train}}$ 的梯度为 0。在梯度下降中，梯度在迭代过程中不断减小，直到梯度为零为止。梯度是一个包含每个维度上偏导数的向量 ，如下所示：

$$
g=\nabla E_{\text{train}}(w)=\left[\frac{\partial E_{\text{train}}}{\partial w_{0}},\frac{\partial E_{\text{train}}}{\partial w_{1}},\cdots,\frac{\partial E_{\text{train}}}{\partial w_{n}}\right]
$$

 归一化梯度 $\hat{g}$ 可以写为：（原梯度向量除以其L2范数）

$$
\hat{g}=\frac{\nabla E_{\text{train}}(w)}{\|\nabla E_{\text{train}}(w)\|}
$$

 在 $\hat{g}$ 方向上设置一个小的步长 $\eta$ ，并相应地更新权重，从而达到最佳点。选择较小的步长很重要，否则算法会振荡并且无法达到最佳点

> 权重既可以初始化为0，也可以设置为随机值，一般从某个正态分布采样

- 梯度下降实现思路

```python
import numpy as np

def gradient_descent(X, y, learning_rate, n_iterations):
    """
    使用梯度下降训练模型。

    参数:
    X (np.array): 特征矩阵
    y (np.array): 目标向量
    learning_rate (float): 学习率
    n_iterations (int): 迭代次数

    返回:
    np.array: 训练好的权重向量
    """
    # 为每个样本添加偏置项 x0 = 1
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    n_samples = X.shape[0]
  
    # 随机初始化权重
    weights = np.random.randn(X_b.shape[1])
  
    for _ in range(n_iterations):
        # 计算预测值
        predictions = X_b.dot(weights)
        # 计算误差
        errors = predictions - y
        # 计算整个数据集的梯度
        gradients = (1/n_samples) * X_b.T.dot(errors)
        # 更新权重
        weights = weights - learning_rate * gradients
  
    return weights
```

梯度下降的缺点之一是在计算梯度时会使用整个训练数据集
批量梯度下降则是计算并更新权重时使用一小部分数据集
随机梯度下降则是随机并均匀地选择一个数据点

- 随机梯度下降实现思路

```python
import numpy as np

def stochastic_gradient_descent(X, y, learning_rate, n_epochs):
    """
    使用随机梯度下降训练模型。

    参数:
    X (np.array): 特征矩阵
    y (np.array): 目标向量
    learning_rate (float): 学习率
    n_epochs (int): 遍历数据集的次数

    返回:
    np.array: 训练好的权重向量
    """
    # 为每个样本添加偏置项 x0 = 1
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    n_samples = X.shape[0]
  
    # 随机初始化权重
    weights = np.random.randn(X_b.shape[1])
  
    for epoch in range(n_epochs):
        for i in range(n_samples):
            # 随机选择一个数据点
            random_index = np.random.randint(n_samples)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
    
            # 计算预测值
            prediction = xi.dot(weights)
            # 计算误差
            error = prediction - yi
            # 计算单个数据点的梯度
            gradient = xi.T.dot(error)
            # 更新权重
            weights = weights - learning_rate * gradient
    
    return weights
```
