# GAT(Graph Attention Network)

## 1. 概括

&emsp;&emsp;本文提出Graph Attention Networks(GATs)，将注意力机制应用到图神经网络中，每一层学习节点每个邻居对其生成新特征的贡献度，按照贡献度大小对邻居特征进行聚合，以此生成节点新特征。

**特点：GATs具有计算复杂度低，适用归纳学习任务的特性。**

&emsp;&emsp;GATs其实是扩展了[GraphSAGE框架](https://zhuanlan.zhihu.com/p/62750137)，**使用了注意力机制的aggregate方法**，从实验效果看显著提升.

## 2. self-attention

本节详细介绍每一次迭代（每一层）中aggregate模块所使用的self-attention机制 （主流的图神经网络架构介绍见另一篇文章[GraphSAGE框架](https://zhuanlan.zhihu.com/p/62750137)）

## 2.1 方法

输入：节点i特征 ![[公式]](https://www.zhihu.com/equation?tex=h_i) , 邻居特征 ![[公式]](https://www.zhihu.com/equation?tex=h_j)

输出：邻居j对节点i生成新特征的贡献度 ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bij%7D)

模型：使用一个简单的前馈神经网络去计算 ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bij%7D) ， 共享参数 ![[公式]](https://www.zhihu.com/equation?tex=W%2C%5Calpha) 通过反向传播学习。

如下图所示：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.nzqe9upd1wd.png)

## 2.2 输入预处理

对 $h_i,h_j$ 线性变换，得到$Wh_i$ 和 $Wh_j$ 。

1. W为参数矩阵，将F维的特征线性组合生成F'维特征。
2. 线性变换的目的是得到更强大的表达

## **2.3 输入层->隐层**

​																				$$e_{ij} = LeakyRelu(\alpha^T[Wh_i]||[Wh_j])$$

1. ![[公式]](https://www.zhihu.com/equation?tex=e_%7Bij%7D) 表示邻居j的特征对i的重要性／贡献度。
2. ![[公式]](https://www.zhihu.com/equation?tex=%7C%7C) 表示将将 $Wh_i$ 和$Wh_j$ 拼接起来，作为神经网络的输入（2F'维）
3. $\alpha$ 为输入层->隐层的参数，因为隐藏只有一个神经元，故是一个2F'维的向量。
4. 激活单元使用Leaky ReLU进行非线性转换

## **2.4 隐层->输出层**

为了使不同邻居的贡献度可以对比，使用softmax归一化，最终得到邻居j对节点i生成新特征的贡献度 ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bij%7D)：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.zxe0qoc6n7h.png)

## **2.5 生成节点新特征**

对节点i的邻居特征按贡献度 $a_{ij}$ 进行加权平均后加一个非线性转换，得到节点i的新特征 

​																			$$h^‘_i = \sigma(\sum\limits_{j\in N_i }{a_{ij}Wh_j})$$