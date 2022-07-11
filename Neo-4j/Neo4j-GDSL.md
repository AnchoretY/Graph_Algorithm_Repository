Neo4j GDSL工具函数库，包含的算法主要分为下面几类：

- 中心算法
- 社区发现算法
- 相似度检测算法
- 寻路算法
- 链接预测算法



### 中心算法

1. #### PageRank算法

   &emsp;&emsp;PageRank算法是一种**根据节点输入链接关系和链接到该节点的节点的重要度**来决定节点在图中的重要度的中心算法。就算公式如下所示：

   <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.n2chnwjxmk.png" alt="image" style="zoom:50%;" />

   &emsp;&emsp;其中，

   - 假设有T1-Tn节点指向节点A
   - d是阻尼系数，可以在0（含）和1（不含）之间设置。通常设置为0.85
   - *C(A)*是A节点的出度

   **使用：**

   &emsp;&emsp;使用流模式格式如下所示：

   ~~~sql
   CALL gds.pageRank.stream(
     graphName: String,
     configuration: Map
   )
   YIELD
     nodeId: Integer,
     score: Float
   ~~~

   - nodeId: 节点Id

   - score： 节点重要度分数

2. #### Betweenness Centrality

   &emsp;&emsp;Betweenness Centrality算法计算图中所有结点对之间的未加权最短路径。每个节点都会根据通过节点的最短路径数量获得分数。更频繁地位于其他节点之间最短路径上的节点将具有更高的中间集中度得分

   > 本算法内存要求较高，实现需要*O(n + m)*空间，并以*O(n\*m)*时间运行，其中*n*是节点数，*m*是图中关系数。



### 社区检测算法

1. #### Louvain

   &emsp;&emsp;Louvain算法是一种基于模块化度(Modularity)的社区发现算法，模块化度是一种衡量衡量社区紧密度的指标。如果一个社区中节点加入到社区使当前社区的模块化度增加，那么节点属于这个社区，

   > 适合大规模社区检测

2. #### LPA

3. #### Triangle Count

4. 

