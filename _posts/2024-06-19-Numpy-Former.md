---
layout: post
header-style: text
title: 用numpy实现一个可以训练的Transformer
tags: AI
author: Suffoquer
catalog: true
preview: 我一直想做这个事情来着，现在终于有时间闲下来写一写了
---

> 我一直想做这个事情来着，现在终于有时间闲下来写一写了。

这个东西的目的是让自己对Transformer有一个更深入的理解，同时也能让一些人不需要装torch或者是llama.cpp这些相关依赖，就能大概运行起来LLaMA，<s>甚至是训练，不过CPU实在是太慢了</s>。

我不打算实现自动求导，而选择手动对每一部分计算梯度，直接写死在代码里。<s>我有点后悔了，手算梯度实在是太痛苦了</s>。

### Linear Layer

首先最开始，我们来实现标准的线性层。

对于一个线性层
$$
y = x \cdot W^T + b\\~\\
W \in \mathbb{R}^{m \times n}, x \in \mathbb{R}^n, b \in \mathbb{R}^{m}, y \in \mathbb{R}^m
$$

我们可以很容易的计算出梯度

$$
{\partial y_{i, j} \over \partial W_{k, l}} = x_{i, l} \cdot \delta_{j, k}\\~\\
{\partial y_{i, j} \over \partial b_{k}} = \delta_{j, k} \\~\\
\delta_{i, j} = \begin{cases}
1 & i = j\\
0 & i \neq j
\end{cases}
$$

写成矩阵形式就是
$$
{\partial y \over \partial W} = x^T \in \mathbb{R}^{n \times m}\\~\\
{\partial y \over \partial b} = [1, 1, \cdots, 1]^T \in \mathbb{R}^{m}
$$


但是实际上在使用的时候，我们往往会把线性层堆叠起来构成深度的网络，所以在算梯度时候需要考虑下游的梯度。

这里我们假设输出$y$之后的下游部分为$f(y)$，那么我们可以用链式法则计算出参数的梯度：
$$
{\partial f(y) \over \partial W} = {\partial f(y) \over \partial y} \cdot {\partial y \over \partial W} = {\partial f(y) \over \partial y} \cdot x^T \\
~\\
{\partial f(y) \over \partial b} = {\partial f(y) \over \partial y} \cdot {\partial y \over \partial b} = {\partial f(y) \over \partial y} \cdot [1, 1, \cdots, 1]^T = \sum_{i=1}^{m} {\partial f(y) \over \partial y_i}
$$

同时，传给上游的梯度为：
$$
{\partial f(y) \over \partial x} = {\partial f(y) \over \partial y} \cdot W
$$

用代码简单实现一下：
```python
def backward(self, grad_output):
    x = self.x
    self.grad_weights = grad_output.T @ x
    self.grad_bias = np.sum(grad_output, axis=0)
    return grad_output @ self.weights
```

### Embedding Layer



