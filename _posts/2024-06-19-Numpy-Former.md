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

### Activation Function

激活函数有很多，比如常见的ReLU：

$$
y = \max(0, x)
$$

```python
class ReLU(Module):
    def __init__(self):
        super().__init__("ReLU")
    
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)
    
    def backward(self, grad_output):
        index = self.x > 0
        grad_input = grad_output * index
        return grad_input
```

这里面比较复杂的是Softmax，这里参考了[这篇文章](https://e2eml.school/softmax.html)。


$$
{\partial{\sigma_k} \over \partial{x_i}} = \left[
\begin{matrix}
\sigma_1(1 - \sigma_1) & -\sigma_1\sigma_2 & \cdots & -\sigma_1\sigma_N\\
-\sigma_2\sigma_1 & \sigma_2(1 - \sigma_2) & \cdots & -\sigma_2\sigma_N\\
\vdots & \vdots & \ddots & \vdots\\
-\sigma_N\sigma_1 & -\sigma_N\sigma_2 & \cdots & \sigma_N(1 - \sigma_N)
\end{matrix}
\right]
$$

```python
# Softmax
def backward(self, grad_output):
    softmax_dim = self.softmax.shape[-1]
    
    identity_shape = (self.softmax.shape[0],) + (1,) * self.softmax.ndim # (batch_size, 1, 1, ..., 1)
    identity = np.identity(softmax_dim, dtype=self.x.dtype)
    identity = np.tile(identity, identity_shape)

    first_term = self.softmax[..., np.newaxis] * identity
    
    expanded = self.softmax[..., np.newaxis]
    second_term = expanded @ self.softmax[..., np.newaxis, :]

    J = first_term - second_term
    
    grad_input = grad_output[..., np.newaxis, :] @ J
    return grad_input.reshape(self.x.shape)
```

但是一般我们也不会直接用Softmax，而是在CrossEntropyLoss中使用，这时候一般会把\(\log\)和和Softmax合并成一个函数，此时梯度会简单很多：
```python
# LogSoftmax
def backward(self, grad_output):
    grad_input = grad_output - self.softmax * np.sum(grad_output, axis=self.axis, keepdims=True)
    return grad_input
```


### Loss Function

我只实现了MSELoss和CrossEntropyLoss，毕竟不是在写完整的框架，这两个肯定是够用了。

MSELoss就很简单了：

$$
\mathcal{L} = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

```python
def loss(self, y, label):
    y = np.asarray(y)
    label = np.asarray(label)
    mse_loss = (y - label) ** 2

    if self.reduction == 'mean':
        return np.mean(mse_loss)
    elif self.reduction == 'sum':
        return np.sum(mse_loss)
    else:
        raise ValueError(f"reduction {self.reduction} is not supported")

def grad(self, y, label):
    num_instances = np.prod(y.shape)
    if self.reduction == 'sum':
        num_instances = 1

    y = np.asarray(y)
    label = np.asarray(label)

    return 2 * (y - label) / num_instances
```

CrossEntropyLoss就不多赘述了，都是类似的东西。

### Training
到这里，我们现有的组件（线性层，激活函数，损失函数）就已经足够训练一个简单的网络了。

拍脑袋想一个最简单的任务

```python
x: [1, 2, 3, 4, 5]
y: [x[0] + x[2] + x[4], max(x[1], x[3])]
```
然后构造一个只有两层的MLP，应该是足够处理这个任务了

```python
xdim = 5
ydim = 2

model = layers.Sequential(
    layers.Linear(xdim, 32),
    layers.ReLU(),
    layers.Linear(32, ydim)
)
```

接着就可以愉快地开始训练了
```python
loss_fn = MSELoss()
learning_rate = 0.01

for epoch in range(100):
    indices = np.random.permutation(data_size)
    average_loss = 0.0
    for i in range(data_size // batch_size):
        batch_indices = indices[i*batch_size:(i+1)*batch_size]
        y_pred = model.forward(x[batch_indices])
        loss, grad = loss_fn(y_pred, y[batch_indices])
        model.backward(grad)
        model.step(learning_rate)
        average_loss += loss
        if (i + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Iteration {i+1}, Loss {average_loss / 50}")
            average_loss = 0.0
```

可以看到loss是在正常下降的
```shell
Epoch 98, Iteration 10, Loss 0.0021909273733704414
Epoch 98, Iteration 20, Loss 0.0021747556001297007
Epoch 98, Iteration 30, Loss 0.002415027525976499
Epoch 99, Iteration 10, Loss 0.002033462687265005
Epoch 99, Iteration 20, Loss 0.0024055140738835513
Epoch 99, Iteration 30, Loss 0.0023627760709594857
Epoch 100, Iteration 10, Loss 0.0021904293227183475
Epoch 100, Iteration 20, Loss 0.002405154366941388
Epoch 100, Iteration 30, Loss 0.002122699694332097
```

最后看一眼训练结果
```shell
Input values:
 [[0.1 0.2 0.3 0.4 0.5]
 [0.2 0.4 0.6 0.8 1. ]]

True values:
 [[0.9 0.4]
 [1.8 0.8]]

Predicted values:
 [[0.92035226 0.38735358]
 [1.80075877 0.76771189]]
```

看起来没什么问题，到这里，我们成功手写了几层神经网络，并且支持SGD的训练了。

### Transformer
现在我们开始正式来实现Transformer的各个部分。

<img src="/img/in-post/2024-06-19/transformer-arch.png" alt="Transformer结构" style="zoom:30%;center;" />

整体来看，需要实现四个部分：Embedding / Encoding，Multi-Head Attention，Feed Forward，Layer Normalization。

#### Embedding / Encoding
Embedding某种意义上就是一个特殊的线性层，只需要把输入的index转化为one-hot向量。这里我就直接按照矩阵乘法的方式来写了，其实应该需要take index来操作，矩阵乘法会占用大量的内存。

Encoding主要是Positional Encoding，原始论文里用的是sine和cosine函数：

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}})\\
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}})
$$

论文也提到说可以用learned positional embedding，结果差不多。BERT里用的是后者，所以无法接受比训练时候更长的序列。

```python
```

#### Multi-Head Attention

<img src="/img/in-post/2024-06-19/attention.png" alt="Attention结构" style="zoom:30%;center;" />

Multi-head Attention实在是太经典了，所以不多介绍了：

$$
\text{Attention}(Q, K, V) = \text{softmax}({QK^T \over \sqrt{d_k}})V\\~\\
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \cdots, \text{head}_h)W^O\\~\\
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

对于Scaled Dot-Product Attention，本质上就是矩阵乘法和Softmax，所以梯度计算很容易，配合上之前实现过的Softmax就行

```python
def backward(self, grad_output):
    grad_v = self.attn.transpose(0, 1, 3, 2) @ grad_output

    grad_output = grad_output @ self.v.transpose(0, 1, 3, 2)
    grad_output = self.softmax.backward(grad_output)

    if self.mask is not None:
        grad_output = np.where(self.mask, grad_output, 0)

    grad_output = grad_output / np.sqrt(d_model)
    grad_q = grad_output @ self.k
    grad_k = grad_output.transpose(0, 1, 3, 2) @ self.q
    
    return grad_q, grad_k, grad_v
```

有了Scaled Dot-Product Attention，Multi-Head Attention的backward只需要按照链式法则调用Attention和四个线性层的backward就行了

```python
def backward(self, grad_output):
    grad_output = self.W_o.backward(grad_output)

    batch_size, seq_len, _ = grad_output.shape
    grad_output = grad_output.reshape(batch_size, seq_len, self.num_heads, self.d).transpose(0, 2, 1, 3)

    grad_q, grad_k, grad_v = self.attention.backward(grad_output)

    grad_q = grad_q.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
    grad_k = grad_k.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
    grad_v = grad_v.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    grad_q = self.W_q.backward(grad_q)
    grad_k = self.W_k.backward(grad_k)
    grad_v = self.W_v.backward(grad_v)

    return grad_q, grad_k, grad_v
```
只要注意一下维度变化，多个head直接concat起来，所以需要一直reshape和transpose。

#### Feed Forward
#### Layer Normalization