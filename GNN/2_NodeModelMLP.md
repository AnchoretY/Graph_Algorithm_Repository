

# NodeModelMLP

&emsp;&emsp;&emsp;NodeModeMLP与NodeMedelMLP非常类似，只是在邻域节点和边的信息完成聚合后，不再采用节点和边的信息直接加和的方式来进行节点信息的更新，而是将节点聚合信息与边聚合信息再通过一个MLP网络来进行映射后，在更新目标节点的信息。

主要流程为：

- 计算聚合标准化系数
- 使用MLP进行边和节点的信息传递
- 聚合节点和边的信息

&emsp;&emsp;下面为二者的不同点主要在两个方面：

> 1. 是否对节点和边的信息进行转化：Additive 在前向传播的第一步，需要对节点和边的特征进行编码表示，
> 2. 信息传递的方式：Additive使用加法进行节点和边的消息传递，MLP使用MLP进行节点和消息传递

&emsp;&emsp;二者进行实现时的区别如下：

~~~python
class NodeModelAdditive(NodeModelBase):
	def __init__(in_channels,out_channels):
		...
    pass
  
 	def forward(x):
    ...
    # ------------节点进行编码映射，这在NodeModelMLP中没有---------
    x = torch.matmul(x, self.weight_node)
    if in_edgedim is not None:
      self.weight_edge = Parameter(torch.Tensor(in_edgedim, out_channels))
      
    # -------------------- 计算聚合系数 ------------------------
    ...
    
    # ------------ 使用加法进行边和节点信息传递，不同点 ------------
    # 获得最终要进行聚合的特征向量，是否包含边特征两种
    x_j = x_j + x_je if edge_attr is not None else x_j	
    
    #--------------------- 聚合节点信息 -------------------------
    # 整合特征信息到节点中,这里需要重点理解 得到(N, C_out)
    x = scatterx_j, edge_index[1], dim_size=x.size(0),reduce=self.aggr)
    ....
~~~

而在NodeModelMLP中，只需要将其改为：

~~~python
class NodeModelAdditive(NodeModelBase):
	def __init__(in_channels,out_channels):
		...
    self.mlp = nn.Linear(in_features, out_channels, bias=bias)
    ...
    def forward(x):
  	# -------------------- 计算聚合系数 ------------------------
    ...
    # ------------ 使用MLP进行边和节点信息传递，不同点 ------------
    if edge_attr is not None:
      x_j = self.mlp(torch.cat([x_j, edge_attr], dim=1))  # size (E, C_out)
      else:
        x_j = self.mlp(x_j)  # size (E, C_out)
        
    #--------------------- 聚合节点信息 -------------------------
    # 整合特征信息到节点中,这里需要重点理解 得到(N, C_out)
    x = scatter_(x_j, edge_index[1], dim_size=x.size(0),reduce=self.aggr)
    ....
~~~



### 完成的NodeModeMLP代码

```python
class NodeModelMLP(NodeModelBase):
    """
        将MLP应用在[node_features, edge_features]上来更新节点特征，节点特征使用标准化的出度
    Note:
        This is currently the same as the :class:`NodeModelAdditive` method,
        for a single layer MLP without non-linearity.
        There is a slight different when `bias` == True: here the bias is applied to messages on each edge
        before doing edge gates, whereas in the above model the bias is applied after aggregation on the nodes.
    """

    def __init__(self, in_channels, out_channels, in_edgedim=None, deg_norm='sm', edge_gate='none', aggr='add',
                 bias=True, mlp_nlay=1, mlp_nhid=32, mlp_act='relu',
                 **kwargs):
        super(NodeModelMLP, self).__init__(in_channels, out_channels, in_edgedim, deg_norm, edge_gate, aggr, **kwargs)

        if in_edgedim is None:
            in_features = in_channels
            # self.mlp = nn.Linear(in_channels, out_channels,
            #                      bias=bias)  # can also have multiple layers with non-linearity
        else:
            in_features = in_channels + in_edgedim
            # self.mlp = nn.Linear(in_channels + in_edgedim, out_channels, bias=bias)

        if mlp_nlay == 1:
            self.mlp = nn.Linear(in_features, out_channels, bias=bias)
        elif mlp_nlay >= 2:
            self.mlp = [nn.Linear(in_features, mlp_nhid, bias=bias)]
            for i in range(mlp_nlay - 1):
                self.mlp.append(activation(mlp_act))
                if i < mlp_nlay - 2:
                    self.mlp.append(nn.Linear(mlp_nhid, mlp_nhid, bias=bias))
                else:
                    # last layer, and we do not apply non-linear activation after
                    self.mlp.append(nn.Linear(mlp_nhid, out_channels, bias=bias))
            self.mlp = nn.Sequential(*self.mlp)

        # self.reset_parameters()

    def reset_parameters(self, initrange=0.1):
        # TODO: this only works for 1-layer mlp
        nn.init.uniform_(self.mlp.weight, -initrange, initrange)
        if self.mlp.bias is not None:
            nn.init.constant_(self.mlp.bias, 0)

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, **kwargs):
        if self.deg_norm == 'none':
            row, col = edge_index
            x_j = x[row]  # size (E, C_in)
            # alternatively
            # x_j = torch.index_select(x, 0, edge_index[0])
        else:

            norm = self.degnorm_const(edge_index, num_nodes=x.size(0), deg=deg,
                                      edge_weight=edge_weight, method=self.deg_norm, device=x.device)
            if self.deg_norm == 'rw' and edge_weight is None:
                x_j = x * norm.view(-1, 1)  # this saves much memory when N << E
                # lift the features to source nodes, resulting size (E, C_out)
                x_j = torch.index_select(x_j, 0, edge_index[0])
            else:
                # lift the features to source nodes, resulting size (E, C_out)
                x_j = torch.index_select(x, 0, edge_index[0])
                x_j = x_j * norm.view(-1, 1)  # norm.view(-1, 1) second dim set to 1 for broadcasting

        if edge_attr is not None:
            assert self.in_edgedim is not None
            x_j = self.mlp(torch.cat([x_j, edge_attr], dim=1))  # size (E, C_out)
        else:
            assert self.in_edgedim is None
            x_j = self.mlp(x_j)  # size (E, C_out)

        # use edge gates
        if self.edge_gate is not None:
            eg = self.edge_gate(x, edge_index, edge_attr=edge_attr, edge_weight=edge_weight)
            x_j = eg * x_j

        # aggregate the features into nodes, resulting size (N, C_out)
        # x_o = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))
        x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))

        return x
```