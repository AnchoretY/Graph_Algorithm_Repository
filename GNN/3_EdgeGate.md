# Gate

&emsp;&emsp;Gate是一种在图神经网络根性过程中为了让模型具有长时记忆而增加的结构



表示对于在信息传递过程中，是否使用门结构来确定保留哪些信息，遗忘哪些信息，从而使图神经网络具有比较长时的记忆。图神经网络中的Gate与GPU中的门结构基本类似。

&emsp;&emsp;主要涉及到的门结构与RNN中的输出门结构类似，计算公式如下：

​										$$r = Sigmod(Wa+Uh)$$

&emsp;&emsp;其中，a、h分别为正反两个方向的向量表示，W、h为线性转化的参数。

#### 代码实现

~~~python
# 获取各个节点的gate
eg = self.edge_gate(x, edge_index, edge_attr)   
# 邻域节点的要聚合的信息经过输出门
x_j = eg * x_j
# 聚合节点信息
x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))
~~~



```python
class EdgeGateProj(nn.Module):
    """
        GRU标准门结构中输出门的门控信号，表达式为:
            r = σ(Wa + Uh)
        其中，
            a为边特征，这里使用的信息双向传递方式，因此Wa = W_in a_in + W_out a_out
            h为邻域节点特征
    """

    def __init__(self, in_channels, in_edgedim=None, bias=False):
        super(EdgeGateProj, self).__init__()

        self.in_channels = in_channels
        self.in_edgedim = in_edgedim

        self.linsrc = nn.Linear(in_channels, 1, bias=False)
        self.lintgt = nn.Linear(in_channels, 1, bias=False)
        if in_edgedim is not None:
            self.linedge = nn.Linear(in_edgedim, 1, bias=False)

        if bias:
            self.bias = Parameter(torch.Tensor(1))  # a scalar bias applied to all edges.
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self, initrange=0.1):
        nn.init.uniform_(self.linsrc.weight, -initrange, initrange)
        nn.init.uniform_(self.lintgt.weight, -initrange, initrange)
        if self.in_edgedim is not None:
            nn.init.uniform_(self.linedge.weight, -initrange, initrange)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index,edge_attr=None):
        x_j = torch.index_select(x, 0, edge_index[0])  # source node features, size (E, C_in)
        x_i = torch.index_select(x, 0, edge_index[1])  # target node features, size (E, C_in)
        
        edge_gate = self.linsrc(x_j) + self.lintgt(x_i)  # size (E, 1)
        if edge_attr is not None:
            assert self.linedge is not None
            edge_gate += self.linedge(edge_attr)
        if self.bias is not None:
            edge_gate += self.bias.view(-1, 1)
        edge_gate = torch.sigmoid(edge_gate)
        return edge_gate
```

