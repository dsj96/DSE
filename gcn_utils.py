'''
Descripttion: 
version: 
Author: ShaojieDai
Date: 2021-01-03 20:49:57
LastEditors: Please set LastEditors
LastEditTime: 2021-09-14 20:07:43
'''
import numpy as np
import scipy.sparse as sp
import torch


''' adj 处理相关的函数'''
def aug_normalized_adjacency(adj):
   adj = adj + sp.eye(adj.shape[0]) #  A + I，A是对称化后的, TODO:避免自身的特征被忽略
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1)) # 按行相加(2708,1)
   d_inv_sqrt = np.power(row_sum, -0.5).flatten() # 各个元素计算-0.5次方
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # 对于无穷大的设置0
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 将其每个元素 作为(2708,2708)上的对角线元素 (D + I)^-1/2
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo() # .dot真实矩阵相乘，非对应位置相乘 TODO:难以将数据限制在我们需要的范围内

def FAME_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1)) # 按行相加(2708,1)
   d_inv_sqrt = np.power(row_sum, -1).flatten() # 各个元素计算-1次方
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # 对于无穷大的设置0
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # 将其每个元素 作为(2708,2708)上的对角线元素 (D)^-1 * adj
   return d_mat_inv_sqrt.dot(adj).tocoo() # .dot真实矩阵相乘，非对应位置相乘

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def fetch_normalization(type): # 获取规范化
   switcher = {
       'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2, A是对称化后的
       'FAMENormAdj': FAME_normalized_adjacency,
   } # dict
   func = switcher.get(type, lambda: "Invalid normalization technique.") # 如果type不是'AugNormAdj'则函数func返回一个字符串"Invalid normalization technique."，正常请款返回一个函数
   return func

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj(adj, normalization="AugNormAdj"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) # 对称化adj
    adj_normalizer = fetch_normalization(normalization) # 返回的是一个函数输入A, 计算A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2 对称的
    adj = adj_normalizer(adj) # A'
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    return adj


'''features 相关函数'''


if __name__ == '__main__':

    adj = nx.adjacency_matrix(G_1)
    adj = preprocess_adj(adj, normalization='AugNormAdj')
    features = torch.eye(G_1.number_of_nodes())

