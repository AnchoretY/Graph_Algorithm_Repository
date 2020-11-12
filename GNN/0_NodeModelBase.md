# NodeModelBase

各种卷积图神经网络的基准类，主要包含了其他类中普遍需要用的的一些类方法。





1. ### degnorm_const



~~~python
@staticmethod
    def degnorm_const(edge_index=None, num_nodes=None, deg=None, edge_weight=None, method='sm', device=None):
        """
        计算归一化常数
        `_sm` 使用对称归一化，"symmetric". 更适合用于无向图.
        `_rw` 使用随即游走归一化(均值),"random walk". 更适合用于有向图.

        Procedure:
            - 检查edge_weight，如果不为None，那么必须同时提供edge_index和num_nodes，计算全部节点的度
            - 如果edge_weighe，如果是None，检查是否已经存在deg(节点的度矩阵):
            	- 如果度矩阵存在，那么忽略edge_index和num_nodes
            	- 如果度矩阵不存在，则必须提供edge_index和num_nodes，并计算全部节点的度
            	
        Input:
            - edge_index (torch.Tensor): COO格式的图关系, (2, E)，long
            - num_nodes (int): 节点数量
            - deg (torch.Tensor): 节点的度,(N,),float
            - edge_weight (torch.Tensor): 边权重,(E,),float
            - method (str): 度标准化方法, choose from ['sm', 'rw']
            - device (str or torch.device): 驱动器编号

        Output:
            - norm (torch.Tensor): 基于节点度和边权重的标准化常数.
                If `method` == 'sm', size (E,);
                if `method` == 'rw' and `edge_weight` != None, size (E,);
                if `method` == 'rw' and `edge_weight` == None, size (N,).

        where
            N: 节点数量
            E: 边数量
        """
        assert method in ['sm', 'rw']

        if device is None and edge_index is not None:
            device = edge_index.device

        if edge_weight is not None:
            assert edge_index is not None, 'edge_index must be provided when edge_weight is not None'
            assert num_nodes is not None, 'num_nodes must be provided when edge_weight is not None'

            edge_weight = edge_weight.view(-1)
            assert edge_weight.size(0) == edge_index.size(1)
						
            calculate_deg = True    # 时候需要计算节点度
            edge_weight_equal = False
        else:
            if deg is None:
                assert edge_index is not None, 'edge_index must be provided when edge_weight is None ' \
                                               'but deg not provided'
                assert num_nodes is not None, 'num_nodes must be provided when edge_weight is None ' \
                                              'but deg not provided'
                edge_weight = torch.ones((edge_index.size(1),), device=device)
                calculate_deg = True
            else:
                # node degrees are provided
                calculate_deg = False
            edge_weight_equal = True

        row, col = edge_index
        # 计算节点度
        if calculate_deg:
            deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes,reduce='add')
				# 节点度标准化
        if method == 'sm':
            deg_inv_sqrt = deg.pow(-0.5)
        elif method == 'rw':
            deg_inv_sqrt = deg.pow(-1)
        else:
            raise ValueError

        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
				
        if method == 'sm':
          	# 采用对称标准化的方式，得到的结果向量为(E,)
            norm = (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col] if not edge_weight_equal # 注意，这里没有直接使用deg_inv_sqrt是因为要乘以权重
                    else deg_inv_sqrt[row] * deg_inv_sqrt[col])  # size (E,)
        elif method == 'rw':
          # 采用随即游走标准化，如果没有边权重矩阵，那么直接输出签名开方的结果，为（N,），否则与上面类似输出为(E,)
            norm = (deg_inv_sqrt[row] * edge_weight if not edge_weight_equal  # size (E,)
                    else deg_inv_sqrt)  # size (N,)
        else:
            raise ValueError

        return norm
~~~





2. ### num_parameters

   

3. ### \__repr__

