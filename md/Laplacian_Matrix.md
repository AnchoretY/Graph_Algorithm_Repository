## 拉普拉斯矩阵
&emsp;&emsp;定义图无向图G(V,E,W)，其中，V为图中的顶点，E为图中的边，W为边上权值构成的矩阵。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ummt1g179o.png" alt="image" width=500 />
### 1.邻接矩阵
&emsp;&emsp;邻接矩阵表示节点之间的关系，是图的结构的表达，常使用A表示。以无权图为例，有连接的两个节点对应的权值为1，没有连接的两个节点之间的权值为0，上面的图对应的邻接矩阵为：
<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.rsnh5fs81q9.png" alt="image" width=500 />


### 1.度矩阵
&emsp;&emsp;节点的度为节点与其他节点连接的权重之和。  
  $$d_i = \displaystyle\sum^{N}_{j=1}{w_{ij}} $$
&emsp;&emsp;度矩阵为各个节点的度值所组成的矩阵，**常用D表示**，上面图的度矩阵为：
<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.97stlbj6y3a.png" alt="image"  width=500 />



