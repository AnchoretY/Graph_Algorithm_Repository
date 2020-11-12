#  NodeModelAdditive

&emsp;&emsp;采用标准的加法运算进行邻域节点消息传递的图神经网络层结构。即采用公式：

​									$$x = x_n +x_e$$

主要的流程为：

- 将节点进行编码表示$X_e = WX_e$
- 将边进行编码表示$X_n = WX_n$
- 计算聚合标准化系数

- 使用加法进行邻域边和节点的信息传递

  

下面将对NodeModelAdditive模型的构建方式进行详细的描述：

1. #### 对节点进行编码表示

   将节点从一种维度的矩阵表示转移到另一种维度的向量表示，具体实现部分代码如下：

   ~~~python
   class NodeModelAdditive(NodeModelBase):
   	def __init__(in_channels,out_channels):
   		...
   		# 节点权重矩阵
       self.weight_node = Parameter(torch.Tensor(in_channels, out_channels))
       ...
    	def forward(x):
       # 将节点特征转化为向量表达， (Node_nums, C_out)        
       x = torch.matmul(x, self.weight_node)
    		...
   ~~~

   > 这里也可以直接使用没有bias的全连接层进行映射

2. #### 对边进行编码表示(可选)

   对于需要对将边信息融入信息传递的图卷积神经网络，可以采用与节点相同的方式进行进行计算。

   ~~~python
   class NodeModelAdditive(NodeModelBase):
   	def __init__(in_edgedim, out_channels):
   		...
   		# 边权重矩阵
       if in_edgedim is not None:
         self.weight_edge = Parameter(torch.Tensor(in_edgedim, out_channels))
       ...
    	def forward(x):
       ...
       # 构建边特征向量(如果存在的话)
       if edge_attr is not None:
         assert self.in_edgedim is not None
         x_je = torch.matmul(edge_attr, self.weight_edge)  # size (E, C_out)
    		...
   ~~~

   

3. #### 节点聚合

   &emsp;&emsp;节点聚合的方式目前主要分为Sum、Mean标准化、Symmetric标准化三种，下面将对这几种方式进行详细的介绍。

   1. **Sum**

      &emsp;&emsp;这种节点聚合的方式比较简单，不需要对节点的度进行标准化。

      <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.hogkzxialkk.png" alt="image" style="zoom:50%;" />

      &emsp;&emsp;实现方式如下：

      ~~~python
      class NodeModelAdditive(NodeModelBase):
      	def __init__():
      		...
       	def forward(x):
          ...
          # 直接使用起始节点特征形成(E, C_out)的起始节点向量矩阵
      	r)
          ...
      ~~~

      > edge_index[0]为所有边的起始节点索引

   2. **Mean**

      &emsp;&emsp;这种聚合方式其本质上是一种标准化后然后在进行聚合，因此将其分为两个部分，第一部分为标准化，第二部分为聚合。

      <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.1mbb6sc1ybf.png" alt="image" style="zoom:50%;" />

      ~~~python
      def degnorm_const(edge_index,edge_weight,num_node,edge_weight_equal):
        """
        	计算节点的平均标准化系数，如果边的权重不同，则注意加权
        """
        row, col = edge_index
        # 计算节点度
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes,reduce='add')
        # 节点度标准化
        deg_inv_sqrt = deg.pow(-1)
        # 负无穷值转变为0
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
      	# 在标准化系数再乘以边的权重(这里与上面的不同点，如果存在边权重不等，那么多乘以边的权重)
        norm = (deg_inv_sqrt[row] * edge_weight if not edge_weight_equal  # size (E,)
                else deg_inv_sqrt)  # (N,)
        return norm
      
      class NodeModelAdditive(NodeModelBase):
      	def __init__(in_channels,out_channels):
      		...
          
       	def forward(x):
          ...
          # 计算各个节点的标准化系数，（E,）或（N,）
          norm = self.degnorm_const(edge_index, num_nodes=x.size(0), deg=deg,
                                    edge_weight=edge_weight, method=self.deg_norm, device=x.device)
          # 将平均标准化系数应用于每个节点的特征
          x_j = x * norm.view(-1, 1)  
         
          # 将每个节点的标准化后的向量表示提升到每个边的源节点的维度 (E, C_out)
          x_j = torch.index_select(x_j, 0, edge_index[0])
          # 获得最终要进行聚合的特征向量，是否包含边特征两种
          x_j = x_j + x_je if edge_attr is not None else x_j
          # 整合特征信息到节点中,这里需要重点理解 得到(N, C_out)
          x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))
          ...
      ~~~

   3. **Symmetric normalization**

      &emsp;&emsp;在GCNN论文中论文中，认为对于节点分类的任务中，图数据中度很大的节点可能并不是很重要的论文，因此不止使用自身的度做归一化，还加入了邻居节点的度做归一化，从而减弱被具有大量邻居节点的对该节点的影响。

      <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.p6tdwyiidm.png" alt="image" style="zoom:33%;" />

      ~~~python
      def degnorm_const(edge_index,edge_weight,num_node,edge_weight_equal):
        """
        	计算节点的平均标准化系数，如果边的权重不同，则注意加权
        """
        row, col = edge_index
        # 计算节点度
        deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes,reduce='add')
        # 节点度标准化
        deg_inv_sqrt = deg.pow(-0.5)
        # 负无穷值转变为0
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
      	# 在标准化系数再乘以边的权重(这里与上面的不同点，如果存在边权重不等，那么多乘以边的权重)
        # 采用对称标准化的方式，得到的结果向量为(E,)
        norm = (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] if not edge_weight_equal else deg_inv_sqrt[row] * deg_inv_sqrt[col])  # size (E,)
        
        return norm
      
      class NodeModelAdditive(NodeModelBase):
      	def __init__(in_channels,out_channels):
      		...
          
       	def forward(x):
          ...
          # 计算各个节点的标准化系数，（E,）或（N,）
          norm = self.degnorm_const(edge_index, num_nodes=x.size(0), deg=deg,
                                    edge_weight=edge_weight, method=self.deg_norm, device=x.device)
          # 将平均标准化系数应用于每个节点的特征
          x_j = x * norm.view(-1, 1)  
         
          # 将每个节点的标准化后的向量表示提升到每个边的源节点的维度 (E, C_out)
          x_j = torch.index_select(x_j, 0, edge_index[0])
          # 获得最终要进行聚合的特征向量，是否包含边特征两种
          x_j = x_j + x_je if edge_attr is not None else x_j
          # 整合特征信息到节点中,这里需要重点理解 得到(N, C_out)
          x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))
          ...
      ~~~



### 完整的NodeModelAdditive代码：

> 下面的代码为上面的代码片段实际使用版本，可能与上面的示意代码略有不同

~~~python
class NodeModelAdditive(NodeModelBase):
    """
    	通过邻域节点的节点和边特征更新节点特征，节点特征表示选用节点的出度
    """

    def __init__(self, in_channels, out_channels, in_edgedim=None, deg_norm='sm', edge_gate='none', aggr='sum',
                 bias=True,
                 **kwargs):
        super(NodeModelAdditive, self).__init__(in_channels, out_channels, in_edgedim, deg_norm, edge_gate, aggr,
                                                **kwargs)
				# 节点权重矩阵
        self.weight_node = Parameter(torch.Tensor(in_channels, out_channels))
				# 边权重矩阵
        if in_edgedim is not None:
            self.weight_edge = Parameter(torch.Tensor(in_edgedim, out_channels))
            
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_node)
        if self.in_edgedim is not None:
            glorot(self.weight_edge)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, deg=None, edge_weight=None, **kwargs):
        
        # 将节点特征转化为向量表达， (Node_nums, C_out)        
        x = torch.matmul(x, self.weight_node)
      
        # 构建边特征向量(如果存在的话)
        if edge_attr is not None:
            assert self.in_edgedim is not None
            x_je = torch.matmul(edge_attr, self.weight_edge)  # size (E, C_out)

        # 为信息传递准备节点特征, 包括信息normalization和合并边缘两部分
        if self.deg_norm == 'none':
            # 直接使用起始节点特征形成(E, C_out)的起始节点向量矩阵          
            x_j = torch.index_select(x, 0, edge_index[0])
        else:
            # 使用节点的度和边权重计算节点的正则化量，（E,）或（N,）
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
				#----------------- 聚合节点+边特征，得到最终新的节点特征--------------
        # 获得最终要进行聚合的特征向量，是否包含边特征两种
        x_j = x_j + x_je if edge_attr is not None else x_j

        # 整合特征信息到节点中,这里需要重点理解 得到(N, C_out)
        x = scatter_(self.aggr, x_j, edge_index[1], dim_size=x.size(0))

        # 添加bias
        if self.bias is not None:
            x = x + self.bias

        return x
      
~~~

