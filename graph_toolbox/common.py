#!




def plot_comunity_graph(partition):
    """
        graph可视化
        Parameters:
        -------------------------
            partition: dict，使用community_louvain进行社区发现的结果
                注：{0: 0,
                    1: 0,
                    2: 1,
                    ...},key表示节点名称，value表示所属于的community
                
    """
    node_color = []
    for node,comunity in partition.items():
        node_color.append(comunity)
    nx.draw_networkx(G, node_color=node_color)
    plt.show()


def view_result(partition):
    """
        查看社区发现结果
        Parameters:
        -------------------------
            partition: dict，使用community_louvain进行社区发现的结果
                注：{0: 0,
                    1: 0,
                    2: 1,
                    ...},key表示节点名称，value表示所属于的community
                
    """
    count = 0
    print("Network has {} comunitys!".format(len(set(partition.values()))))
    print("The info of comunitys are as fllow:")
    for com in set(partition.values()) :
        count = count + 1
        list_nodes = [nodes for nodes in partition.keys()
                                 if partition[nodes] == com]
        print("Comunitys {} is: {}".format(count,list_nodes))
    
    # 可视化
    plot_comunity_graph(partition)




def plot_comunity_graph2(cluster_result):
    """
        graph可视化
        Parameters:
        -------------------------
            cluster_result: dict_valueiterator，使用Network进行分类的结果
                注：这种类型的数据只能进行遍历访问，其遍历访问每次返回一个Set，其中元素为属于一列的节点编号
                
    """
    node_color = []
    for i,data in enumerate(cluster_result):
        for node_number in data:
            node_color.append(i)
    
    nx.draw_networkx(G,node_color=node_color)
    plt.show()

    


def view_result2(cluster_result):
    """
        查看社区发现结果
        Parameters:
        -------------------------
            cluster_result: dict_valueiterator，使用Network进行分类的结果
                注：这种类型的数据只能进行遍历访问，其遍历访问每次返回一个Set，其中元素为属于一列的节点编号
                
    """
    cluster_result = list(cluster_result)
    print("Network has {} comunitys!".format(len(cluster_result)))
    print("The info of comunitys are as fllow:")
    for i,cluster in enumerate(cluster_result):
        print("Comunitys {} is: {}".format(i,cluster))
    
    # 可视化
    plot_comunity_graph2(cluster_result)
          
view_result2(list(partition_lpa))
