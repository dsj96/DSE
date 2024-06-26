'''
Descripttion: 
version: 
Author: ShaojieDai
Date: 2021-05-18 21:08:27
LastEditors: sueRimn
LastEditTime: 2021-05-18 21:09:29
'''
import networkx as nx

def find123Nei(G, node):
    nodes = list(nx.nodes(G))
    nei1_li = []
    nei2_li = []
    nei3_li = []
    for FNs in list(nx.neighbors(G, node)):  # find 1_th neighbors
        nei1_li .append(FNs)

    for n1 in nei1_li:
        for SNs in list(nx.neighbors(G, n1)):  # find 2_th neighbors
            nei2_li.append(SNs)
    nei2_li = list(set(nei2_li) - set(nei1_li))
    if node in nei2_li:
        nei2_li.remove(node)

    for n2 in nei2_li:
        for TNs in nx.neighbors(G, n2):
            nei3_li.append(TNs)
    nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
    if node in nei3_li:
        nei3_li.remove(node)

    return nei1_li, nei2_li, nei3_li


if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from(list(range(1, 8)))
    G.add_edges_from([(1, 2), (1, 3), (1, 5), (1, 4), (2, 8), (2, 6), (3, 6), (4, 7)])

    neighbors = find123Nei(G, 1)
    print(neighbors[0])  # 输出节点的一阶邻居
    print(neighbors[1])  # 输出节点的二阶邻居
    print(neighbors[2])  # 输出节点的三阶邻居

