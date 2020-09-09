## 拉普拉斯矩阵
&emsp;&emsp;定义图无向图G(V,E,W)，其中，V为图中的顶点，E为图中的边，W为边上权值构成的矩阵。  

<div align=center><img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ummt1g179o.png" alt="image" width=500 /></div>

### 1.邻接矩阵
&emsp;&emsp;邻接矩阵表示节点之间的关系，是图的结构的表达，常使用A表示。以无权图为例，有连接的两个节点对应的权值为1，没有连接的两个节点之间的权值为0，上面的图对应的邻接矩阵为：  
<div align=center><img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.rsnh5fs81q9.png" alt="image" width=500 /></div>


### 2.度矩阵
&emsp;&emsp;节点的度为节点与其他节点连接的权重之和。  
  <div align=center><img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.svdtf2ncq8f.png" alt="image" width=100 /></div>



### 3.拉普拉斯矩阵
&emsp;&emsp;拉普拉斯矩阵的定义为
$$ L = D - A $$


### 4.正则拉普拉斯矩阵
&emsp;&emsp;在现实中更加常用的拉普拉斯矩阵形式为正则拉普拉斯矩阵，定义为:
$$L^{sym}:=D^{-1/2}LD^{-1/2}=I-D^{-1/2}AD^{-1/2}$$
&emsp;&emsp;矩阵中元素的计算法是如下：  
<div align=center><img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.bk3fww2ra6d.png" alt="image"  width=500 /></div>

