# GNN

&emsp;&emsp;这部分将对主要的图神经网络结构进行复现，并对图神经网络中常见函数功能进行讲解。

### 概述

&emsp;&emsp;目前，图神经网络主要

包含了依赖库：

- torch_scatter
- Torch_



### 常用函数

#### scatter库

&emsp;&emsp;scatter库实现的功能为索引上相同的值src矩阵中对应的位置的元素进行加和，常用语计算相邻节点的信息传递过程。函数的参数包括：

- **src** (`Tensor`) – 要进行聚合的Tensor
- **index** (`Tensor`) – 进行元素聚合指定的索引，索引值相同的将聚合在一起
- **dim** ([`int`](https://docs.python.org/3/library/functions.html#int)) – 索引对齐的方向 (default: `-1`)
- **out** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[`Tensor`]) – 目标tensor，很少用.
- **dim_size** ([`Optional`](https://docs.python.org/3/library/typing.html#typing.Optional)[[`int`](https://docs.python.org/3/library/functions.html#int)]) – 如果没有指定`out`,自动使用`dim_size`的维度维度创建输出，如果dim_size没有指定，那么输出的Tensor维度将不能小于`index.max()+1`
- **reduce** ([`str`](https://docs.python.org/3/library/stdtypes.html#str)) – 指定scatter采用的聚合操作类型 (`"sum"`, `"mean"`, `"min"` or `"max"`). (默认: `"sum"`)

{% note  warning %}
注意在新版本的scatter_add中不在具有fill_value参数，版本更新时API变动较大，注意查看对应版本的官方文档
{% endnote %}

&emsp;&emsp;scatter(....,reduce='sum')的工作原理如下图所示。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.czlizeelbx4.png" alt="image" style="zoom:30%;" />

~~~python
>> index  = torch.tensor([[2,1],[1,3],[0,2],[3,0],[3,1],[3,2]])
tensor([[2, 1],
        [1, 3],
        [0, 2],
        [3, 0],
        [3, 1],
        [3, 2]])
>> src = torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]).float()
tensor([[ 1.,  2.],
        [ 3.,  4.],
        [ 5.,  6.],
        [ 7.,  8.],
        [ 9., 10.],
        [11., 12.]])
>> output = torch.zeros((4,4))
tensor([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]])
>> torch_scatter.scatter(src,index,0,reduce='sum')
tensor([[ 5.,  8.],
        [ 3., 12.],
        [ 1., 18.],
        [27.,  4.]])
>> torch_scatter.scatter(src,index,0,out=output,reduce='sum')
tensor([[ 5.,  8.,  0.,  0.],
        [ 3., 12.,  0.,  0.],
        [ 1., 18.,  0.,  0.],
        [27.,  4.,  0.,  0.]])
  
~~~

1. GNN（Graph Neutual Network）

2. Edge gete

3. GATs（Graph Attention Networks）

   





