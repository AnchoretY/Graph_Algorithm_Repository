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

**代码实现**：

~~~python
claclass NodeModelAttention(nn.Module):
    
    def __init__(self, in_channels, out_channels,att_act='none', att_dir='in'):
        
        super(NodeModelAttention, self).__init__()
        self.weight = Parameter(torch.Tensor(in_channels,out_channels)) # 线性变换参数矩阵

    def forward(self,x,edge_index):
        # 线性变化（N,C_out）
        x = torch.matmul(x,weight)
       
~~~

## **2.3 输入层->隐层**

​																				$$e_{ij} = LeakyRelu(\alpha^T[Wh_i]||[Wh_j])$$

1. ![[公式]](https://www.zhihu.com/equation?tex=e_%7Bij%7D) 表示邻居j的特征对i的重要性／贡献度。
2. ![[公式]](https://www.zhihu.com/equation?tex=%7C%7C) 表示将将 $Wh_i$ 和$Wh_j$ 拼接起来，作为神经网络的输入（2F'维）
3. $\alpha$ 为输入层->隐层的参数，因为隐藏只有一个神经元，故是一个2F'维的向量。
4. 激活单元使用Leaky ReLU进行非线性转换

**代码实现：**

~~~python
class NodeModelAttention(nn.Module):
    
    def __init__(self, in_channels, out_channels,aggr='add', att_dir='in'):
        
        super(NodeModelAttention, self).__init__()
        self.weight = Parameter(torch.Tensor(in_channels,out_channels)) # 线性变换参数矩阵
        self.att_weight = Parameter(torch.Tensor(1, 2*out_channels))   # 注意力权重矩阵
        self.att_act = nn.LeakyReLU() # 使用的非线性激活函数，relu、LeakyRelu等
        self.att_dir = att_dir # 注意力进行softmax计算时的方向
        self.aggr = aggr # 邻域节点进行信息整合的方式
        

    def forward(self,x,edge_index):
        # 线性变化（N,C_out）
        x = torch.matmul(x,self.weight)
        
        # 输入层到隐层，使用非线性激活函数进行转化
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])
        x_data = torch.cat([x_j, x_i], dim=-1) * self.att_weight   # (N,C_out)
        alpha = self.att_act(x_data.sum(dim=-1))  # (N,1)

~~~



## **2.4 隐层->输出层**

为了使不同邻居的贡献度可以对比，使用softmax归一化，最终得到邻居j对节点i生成新特征的贡献度 ![[公式]](https://www.zhihu.com/equation?tex=a_%7Bij%7D)：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.zxe0qoc6n7h.png)

**代码实现：**

~~~python
class NodeModelAttention(nn.Module):
    
    def __init__(self, in_channels, out_channels,aggr='add', att_dir='in'):
        
        super(NodeModelAttention, self).__init__()
        self.weight = Parameter(torch.Tensor(in_channels,out_channels)) # 线性变换参数矩阵
        self.att_weight = Parameter(torch.Tensor(1, 2*out_channels))   # 注意力权重矩阵
        self.att_act = nn.LeakyReLU() # 使用的非线性激活函数，relu、LeakyRelu等
        self.att_dir = att_dir # 注意力进行softmax计算时的方向
        

    def forward(self,x,edge_index):
        # 线性变化（N,C_out）
        x = torch.matmul(x,self.weight)
        
        # 输入层到隐层，使用非线性激活函数进行转化
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])
        x_data = torch.cat([x_j, x_i], dim=-1) * self.att_weight   # (N,C_out)
        alpha = self.att_act(x_data.sum(dim=-1))  # (N,1)

        # 使用softmax将各个节点的权重和规整到1
        if self.att_dir == 'out':
            alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))   # 所有out（以模型为起点）的节点对注意力系数进行softmax
        else:
            alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))    # 所有in（以模型为重点）的节点对注意力系数进行softmax
            
            
def softmax(src, index, num_nodes=None):
    """
    按照索引指定方式进行softmax，索引值相同的为一组，每组和为1
    """

    if num_nodes is None:
        num_nodes = index.max().item() + 1
    print(index)
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    print(out)
    # fill_value here above is crucial for correct operation!!
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out

~~~



## **2.5 生成节点新特征**

对节点i的邻居特征按贡献度 $a_{ij}$ 进行加权平均后加一个非线性转换，得到节点i的新特征 

​																			$$h^‘_i = \sigma(\sum\limits_{j\in N_i }{a_{ij}Wh_j})$$

**代码实现：**

~~~python
class NodeModelAttention(nn.Module):
    
    def __init__(self, in_channels, out_channels,aggr='add', att_dir='in'):
        
        super(NodeModelAttention, self).__init__()
        self.weight = Parameter(torch.Tensor(in_channels,out_channels)) # 线性变换参数矩阵
        self.att_weight = Parameter(torch.Tensor(1, 2*out_channels))   # 注意力权重矩阵
        self.att_act = nn.LeakyReLU() # 使用的非线性激活函数，relu、LeakyRelu等
        self.att_dir = att_dir # 注意力进行softmax计算时的方向
        self.aggr = aggr # 邻域节点进行信息整合的方式
        

    def forward(self,x,edge_index):
        # 线性变化（N,C_out）
        x = torch.matmul(x,self.weight)
        
        # 输入层到隐层，使用非线性激活函数进行转化
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])
        x_data = torch.cat([x_j, x_i], dim=-1) * self.att_weight   # (N,C_out)
        alpha = self.att_act(x_data.sum(dim=-1))  # (N,1)

        # 使用softmax将各个节点的权重和规整到1
        if self.att_dir == 'out':
            alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))   # 所有out（以模型为起点）的节点对注意力系数进行softmax
        else:
            alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))    # 所有in（以模型为重点）的节点对注意力系数进行softmax

        # 使用attention系数来调节各个节点的信息比重
        x_j = x_j * alpha.view(-1, 1)
        x = scatter(x_j,edge_index[1],dim_size=x.size(0),dim=0,reduce=self.aggr)
        return x
~~~



## **3. multi-head attention**

&emsp;&emsp;因为只计算一次attention，很难捕获邻居所有的特征信息，[《Attention is all you need](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762)》论文中进一步完善了attention机制，提出了multi-head attention ，其实很简单，就是重复做多次attention计算），如下图所示：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.43jihg03qy5.png)

> 图中每种颜色的线条分别代表不同颜色的attention

&emsp;&emsp;本文也使用了multi-head attention：学习K个不同的attention，对应参数 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ek_%7Bij%7D%2CW%5Ek+) ，然后在生成节点i的新特征时拼接起来：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.bmqa2t22a5v.png)

&emsp;&emsp;如果在整个图神经网络的最后一层，使用平均替代拼接，得到节点最终的embedding：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.wfy4pn5420i.png)

**代码实现：**

&emsp;&emsp;多头注意力图神经网络代码实现中与一般地自注意力机制的代码实现主要区别如下：

1. weight 参数维度不同

   ~~~python
   # --------------- self-attention ---------
   # __init__
   self.att_weight = Parameter(torch.Tensor(in_channels, out_channels)   # 线性变化的权重矩阵
   # forward
   x = torch.matmul(x,self.weight)
   
   
   # -------------- muti-attention ----------
   # __init__
   self.weight = Parameter(torch.Tensor(in_channels,nhead*out_channels)) # 线性变换参数矩阵
   # forward
   x = torch.matmul(x,self.weight).view(-1, self.nheads, self.out_channels_1head)
   ~~~

2. Att_weight维度不同

   ~~~python
   # --------------- self-attention ---------
   self.att_weight = Parameter(torch.Tensor(1,  2 * self.out_channels))
   
   
   # -------------- muti-attention ----------
   self.att_weight = Parameter(torch.Tensor(1, self.nheads, 2 * self.out_channels)
   ~~~

3. 使用attention系数来调节准备聚合到节点的向量信息时，增加了调整系数形状

   ~~~python
   # --------------- self-attention ---------
   x_j = x_j * alpha.view(-1, 1)
   
   # -------------- muti-attention ----------
   x_j = x_j * alpha.view(-1, self.nheads, 1)
   ~~~

4. 最后对各个头输出的结果取平均（也可加和）

   ~~~
   x = x.mean(dim=1)
   ~~~

**完整代码：**

~~~python
class MutiAttention(nn.Module):
    
    def __init__(self, in_channels, out_channels,nheads=1,aggr='add', att_dir='in'):
        
        super(MutiAttention, self).__init__()
        self.weight = Parameter(torch.Tensor(in_channels,nheads*out_channels)) # 线性变换参数矩阵
        self.att_weight = Parameter(torch.Tensor(1, nheads, 2*out_channels))   # 注意力权重矩阵
        self.att_act = nn.LeakyReLU() # 使用的非线性激活函数，relu、LeakyRelu等
        self.att_dir = att_dir # 注意力进行softmax计算时的方向
        self.aggr = aggr # 邻域节点进行信息整合的方式
        self.nheads = nheads
        self.out_channels = out_channels
        

    def forward(self,x,edge_index):
        # 线性变化（N,nheads*C_out）
        x = torch.matmul(x,self.weight).view(-1, self.nheads, self.out_channels)
        
        # 输入层到隐层，使用非线性激活函数进行转化
        x_j = torch.index_select(x, 0, edge_index[0])
        x_i = torch.index_select(x, 0, edge_index[1])
        x_data = torch.cat([x_j, x_i], dim=-1) * self.att_weight   # (N,nheads,2*C_out)
        alpha = self.att_act(x_data.sum(dim=-1))  # (N,1)

        # 使用softmax将各个节点的权重和规整到1
        if self.att_dir == 'out':
            alpha = softmax(alpha, edge_index[0], num_nodes=x.size(0))   # 所有out（以模型为起点）的节点对注意力系数进行softmax
        else:
            alpha = softmax(alpha, edge_index[1], num_nodes=x.size(0))    # 所有in（以模型为重点）的节点对注意力系数进行softmax

        # 使用attention稀疏来调节各个节点的信息比重
        x_j = x_j * alpha.view(-1, self.nheads, 1)
        x = scatter_add(x_j,edge_index[1],dim_size=x.size(0),dim=0)
        return x
~~~

