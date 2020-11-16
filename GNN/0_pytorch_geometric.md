## 2. 基本概念介绍

### 2.1 Data Handling of Graphs 图形数据处理

&emsp;&emsp;图（Graph）是描述实体（节点）和关系（边）的数据模型。在Pytorch Geometric中，图被看作是[torch_geometric.data.Data](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)的实例，并拥有以下属性：

| 属性              | 描述                                                         |
| ----------------- | ------------------------------------------------------------ |
| `data.x`          | 节点特征，维度是`[num_nodes, num_node_features]`。           |
| `data.edge_index` | 维度是`[2, num_edges]`，描述图中节点的关联关系，每一列对应的两个元素，分别是边的起点和重点。数据类型是`torch.long`。需要注意的是，`data.edge_index`是定义边的节点的张量（tensor），而不是节点的列表（list）。 |
| `data.edge_attr`  | 边的特征矩阵，维度是`[num_edges, num_edge_features]`         |
| `data.y`          | 训练目标（维度可以是任意的）。对于节点相关的任务，维度为`[num_nodes, *]`；对于图相关的任务，维度为`[1,*]`。 |
| `data.position`   | 节点位置矩阵（Node position matrix），维度为`[num_nodes, num_dimensions]`。 |

&emsp;&emsp;下面是一个简单的例子：

&emsp;&emsp;首先导入需要的包：

```python
import torch
from torch_geometric.data import Data
```

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.4gbuna6oib7.png)

&emsp;&emsp;接着定义边，下面两种定义方式是等价的。第二种方式可能更符合我们的阅读习惯，但是需要注意的是此时应当增加一个`edge_index=edge_index.t().contiguous()`的操作。此外，由于是无向图，虽然只有两条边，但是我们需要四组关系说明来描述边的两个方向。

> **在torch_grometric中，无向图需要将每条边写正反两次，实际边的数量为data.edge_nums/2**

```python
## 法1
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index)

## 法2
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
data = Data(x=x, edge_index=edge_index.t().contiguous())
1234567891011
```

&emsp;&emsp;可以得到：

```python
>>> Data(edge_index=[2, 4], x=[3, 1])
```

&emsp;&emsp;同时，Data对象提供了一些很实用的函数：

```python
print('data\'s keys: {}'.format(data.keys))
print('-'*5)
for key, item in data:
    print("{} found in data".format(key))
print('-'*5)  
print('Does data has attribute \'edge_attr\'? {}'.format('edge_attr' in data))
print('data has {} nodes'.format(data.num_nodes))
print('data has {} edges'.format(data.num_edges))
print('The nodes in data have {} feature(s)'.format(data.num_node_features))
print('Does data contains isolated nodes? {}'.format(data.contains_isolated_nodes()))
print('Does data contains self loops? {}'.format(data.contains_self_loops()))
print('is data directed? {}'.format(data.is_directed()))
print(data['x'])
12345678910111213
```

&emsp;&emsp;输出：

```
data's keys: ['x', 'edge_index']
-----
edge_index found in data
x found in data
-----
Does data has attribute 'edge_attr'? False
data has 3 nodes
data has 4 edges
The nodes in data have 1 feature(s)
Does data contains isolated nodes? False
Does data contains self loops? False
is data directed? False
tensor([[-1.],
        [ 0.],
        [ 1.]])
123456789101112131415
```

同样可以在GPU上运行data：

```python
device = torch.device('cuda')
data = data.to(device)
```

### 2.4 Data Transforms 数据转换

[torch_geometric.transforms.Compose](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.Compose)提供了数据转换的方法，可以方便用户将数据转换成既定的格式或者用于数据的预处理。在之前使用torchvision处理图像时，也会用到数据转换的相关方法，将图片转换成像素矩阵，这里的数据转换就类似torchvision在图像上的处理。

### 2.5 Learning Methods on Graphs——the first graph neural network 搭建我们的第一个图神经网络

下面我们来尝试着搭建我们的第一图神经网络。关于图神经网络，可以看一下这篇博客——[GRAPH CONVOLUTIONAL NETWORKS](http://tkipf.github.io/graph-convolutional-networks/)。

**数据集准备**

我们使用的是Cora数据集。

```python
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='./data/Cora', name='Cora')
print(dataset)
123
```

输出：

```
Cora()
1
```

**搭建网络模型**

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
12345678910111213141516171819
```

模型的结构包含两个[GCNConv](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv)层，选择ReLU作为非线性函数，最后通过softmax输出分类结果。

**模型训练和验证**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
_, pred = model(data).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
123456789101112131415161718
```

输出：

```
Accuracy: 0.8120
```

## 3. CREATING MESSAGE PASSING NETWORKS 建立消息传递网络

&emsp;&emsp;将卷积神经网络中的“卷积算子”应用到图上面，**核心在于neighborhood aggregation机制，或者说是message passing的机制**。**Aggregate Neighbours**，核心思想在于基于局部网络连接来生成Node embeddings（Generate node embeddings based on local network neighborhoods）。如下面这个图：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.0vjvhcq1bzf.png)

&emsp;&emsp;例如图中节点A的embedding决定于其邻居节点{ B , C , D } \{B,C,D\}{*B*,*C*,*D*}，而这些节点又受到它们各自的邻居节点的影响。**图中的“黑箱”可以看成是整合其邻居节点信息的操作，它有一个很重要的属性——其操作应该是顺序（order invariant）无关的**，如求和、求平均、求最大值这样的操作，可以采用神经网络来获取。这样顺序无关的聚合函数符合网络节点无序性的特征，当我们对网络节点进行重新编号时，我们的模型照样可以使用。