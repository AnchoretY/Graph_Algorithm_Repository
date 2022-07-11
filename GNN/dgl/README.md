### 基本操作

1. networkx图转dgl图

```python
# 参数node_attrs指定要转换到dgl graph中的属性
dgl_graph = dgl.from_networkx(graph,node_attrs=['embedding'])
```

2. 获取属性值

```python
# 获取节点属性
dgl_graph.ndata["embedding"]

# 获取边属性值
dgl_graph.edata["embedding"]
```

3. 获取节点和边的连接信息

```python
# 获取节点编号信息，Tensor
dgl_graph.nodes()   #tensor([0, 1, 2, 3, 4, 5])

# 获取边编号信息，两个Tensor
edge_start,end_start = dgl_graph.edges() #(tensor([0, 0, 1, 1, 2, 3]), tensor([5, 1, 3, 2, 3, 4]))
```

4. 获取节点和边数量

```python
# 获取节点数量
dgl_graph.num_nodes()
# 获取边数量
dgl_graph.num_edges()
```

5. 查询节点度

~~~python
# 查询节点编号为0的出度
dgl_graph.out_degrees(0)
# 查询节点编号为0的入度
dgl_graph.in_degrees(0)
~~~



### 数据管道

#### DGLDataset

dgl中包含基于pytorch dataset的DGLDataset类，用于作为自定义数据集的父类。

DGLDataset初始化必须实现的方法有：

- `__init__`:初始化
- `__getitem__`： 根据索引指定元素
- `__len__`：获取数据集数据量(必须实现，否则无法使用dataloader)

```jsx
import dgl
import networkx as nx
from dgl.data import DGLDataset
from util.common_helper import read_pickle

class MyDataset(DGLDataset):
    """ 用于在DGL中自定义图数据集的模板：

    Parameters
    ----------
    url : str
        下载原始数据集的url。
    raw_dir : str
        指定下载数据的存储目录或已下载数据的存储目录。默认: ~/.dgl/
    save_dir : str
        处理完成的数据集的保存目录。默认：raw_dir指定的值
    force_reload : bool
        是否重新导入数据集。默认：False
    verbose : bool
        是否打印进度信息。
    """
    def __init__(self,data_path,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        super(MyDataset, self).__init__(name='dataset_name',
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)
        
        self.file_l = get_file_list(data_path)[:10]

    def __getitem__(self, idx):
        graph =read_pickle(self.file_l[idx])
        graph = dgl.from_networkx(graph,node_attrs=['embedding'])
        graph.edata["embedding"] = torch.ones(graph.num_edges(),4)
        return graph

    def __len__(self):
        # 数据样本的数量
        return len(self.file_l)
```

#### GraphDataloader

dgl中使用基于pytorch的GraphDataloader来基于dataset的批量数据样本获取，主要需要重写的函数为 **collate_fn，在该函数中定义如何将单个样本合并成batch sample。**

```jsx
import torch
import numpy as np
from dgl.dataloading import GraphDataLoader

def collate_fn(batch):
    graphs = dgl.batch([e[0] for e in batch])   #会形成一个大的图，图中包含多个无关联的小图
    nodes_feats = graphs.ndata["embedding"]
    edge_feats = graphs.edata["embedding"]
    return graphs,nodes_feats,edge_feats

dataset = MyDataset("data/train_json_block_embeding_graph_2/")
dataloader = GraphDataLoader(dataset,batch_size=5,shuffle=True,collate_fn=collate_fn)
```

在collate_fn常使用 **dgl.batch 将多个单个的graph合并成为一个由多个不相连的小图组成的大图**，作为batch sample，后续获得batch sample上的节点特征、边特征等都可以基于形成的大图上进行。