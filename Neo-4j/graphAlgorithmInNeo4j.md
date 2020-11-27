
常见社区检测算法总结
![](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.mysknyn8zdk.png)


### 1.标签传播算法
特点：
  1. **单单使用网络结构进行社区的发现**，不需要预定义一些目标函数和先验知识
  2. 线性的时间来进行社区发现  
  
发现过程：
  - 节点使用不同的社区标签进行初始化
  - 标签通过网络进行迭代传播
    - 在每轮传播中，节点将他的社区标签更新为它的邻居中最多属于的社区
  - 最终当所有节点所属于的社区与他的邻居中最多属于的社区全都相同时或达到用户设定的迭代轮数时，模型收敛 

> 在LPA中可以手动为一些节点设置初始时的标签，因此可以是能够进行**半监督学习**的。

#### Cyper调用
1. 创建子图
~~~mysql
# 创建子图
CALL gds.graph.create(
    'myGraph',                                # 创建的子图名称
    'User',                                   # 节点
    'FOLLOW',                                 # 关系
    {
        nodeProperties: 'seed_label',          # 节点属性
        relationshipProperties: 'weight'       # 关系属性
    }
)
~~~

2. 调用LPA进行结果社区聚类
~~~ mysql
CALL gds.labelPropagation.stream('myGraph')
YIELD nodeId, communityId AS Community
RETURN gds.util.asNode(nodeId).name AS Name, Community
ORDER BY Community, Name
~~~
&emsp;&emsp;除了上面这种只使用基本的形式进行社区聚类，还可以通过制定部分节点的初始标签进行半监督学习，以及指定边的权重进行社区发现。
~~~mysql
# 半监督学习
CALL gds.labelPropagation.stream('myGraph', { seedProperty: 'seed_label' })
YIELD nodeId, communityId AS Community
RETURN gds.util.asNode(nodeId).name AS Name, Community
ORDER BY Community, Name


# 指定关系权重进行标签传播
CALL gds.labelPropagation.stream('myGraph', { relationshipWeightProperty: 'weight' })
YIELD nodeId, communityId AS Community
RETURN gds.util.asNode(nodeId).name AS Name, Community
ORDER BY Community, Name
~~~






