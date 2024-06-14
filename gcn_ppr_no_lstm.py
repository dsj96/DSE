'''
Descripttion: 
version: 
Author: ShaojieDai
Date: 2021-05-08 23:06:23
LastEditors: sueRimn
LastEditTime: 2021-05-18 22:28:03
'''
from torch.utils.data import Dataset, DataLoader


import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot # make_dot(prediction)
from torch.autograd import Variable
from tensorboardX import SummaryWriter


# third party
import math
import numpy as np
import time
import itertools
import os # os.path.isfile('test.txt')
import random

# define myself
from args import parse_args
from utils import *
from gcn_models import get_model,STNN,COMBINE_MODEL
from gcn_utils import preprocess_adj
from walk import RWGraph, find_positive_samples


def objective_rw( output, negk, adj_weight_dict, vocab_list, word_freqs):

    loss_term = 0
    for idx,cur_embedding in enumerate(output):
        negative_list = []
        cur_adj = adj_weight_dict[idx] # {    0: {1: {'weight': 0.7310585786300049}, 131: {'weight': 0.7310585786300049}}     }
        while( len(set(negative_list) - set(cur_adj)) < negk ): # TODO: 可能出现死循环，几率很小
            negative_list = random.choices(population=vocab_list, weights=word_freqs, k=negk)

        positive_embedding = output[cur_adj] # pos_num*dim
        # weight_positive_tensor = torch.tensor([item["weight"] for item in list(cur_adj.values())],dtype=torch.float32).unsqueeze(0) # (1, pos_num)

        negative_embedding = output[negative_list] # neg_num*dim

        cur_loss_term =  - torch.sum(F.logsigmoid(torch.cosine_similarity( positive_embedding, cur_embedding, dim=-1) )) \
                         - torch.sum(torch.log( 1. - torch.sigmoid(torch.cosine_similarity(negative_embedding, cur_embedding, dim=-1 ) ) ))

        loss_term = loss_term + cur_loss_term
    return loss_term



'''1.自己编写的'''
class My_dataset(Dataset):
    def __init__(self, LSTM_train_records_input, LSTM_train_records_output):
        '''max_seq_len=-10 negative LSTM_train_records_input= 3 layer list'''
        super().__init__()
        self.LSTM_train_records_input   = LSTM_train_records_input
        self.LSTM_train_records_output  = LSTM_train_records_output
        # self.max_seq_len                = max_seq_len
        self.length = len(LSTM_train_records_input)
        self.src,  self.trg = [], []
        for i in range(self.length):
            self.src.append(self.LSTM_train_records_input[i])
            self.trg.append(self.LSTM_train_records_output[i])

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return len(self.src)

def collate_fn(batch):
    batch_num = len(batch)
    batch_src = list()
    batch_trg = list()
    for i in range(batch_num):
        cur_batch_src = [item for item in batch[i][0]]
        batch_src.append(cur_batch_src)
        cur_batch_trg = [item for item in batch[i][1]]
        batch_trg.append(cur_batch_trg)
    return batch_src, batch_trg


def row_normalize(output_embedding):
    """对一个二维矩阵进行行标准化"""
    row_sum = torch.sum(output_embedding,dim=1)
    r_inv = np.power(row_sum.detach().numpy(), -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(torch.tensor(r_inv))

    return r_mat_inv.matmul(output_embedding)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()


    '''1.都是过程文件，暂时没有到形成index G的阶段'''
    '''加上check_set的目的主要是为了检查user_set和poi_set是否有交集，即poi和user是否有相同id的情况 interset_set原来=13'''
    user_set, poi_set = load_training_data_check_set(args.input_path+'train_checkin_file.txt', args.input_path+'friendship_file.txt', istag=False, isweek=False, hour_interval=4, edge_interval = args.theta)

    u_p_edges, p_p_edges, u_u_edges = load_training_data(args.input_path+'train_checkin_file.txt', args.input_path+'friendship_file.txt', istag=False, isweek=False, hour_interval=4, edge_interval = args.theta, interset_set=user_set&poi_set)
    history = read_history_check_set(args.input_path+'train_checkin_file.txt', interset_set=user_set&poi_set)
    poi_longi_lanti = gen_poi_coordinate(args.input_path+'train_checkin_file.txt', interset_set=user_set&poi_set)

    G = get_G_from_edges_chect_set(u_p_edges, p_p_edges, u_u_edges, history, args.epsilon)
    node_type = gen_node_type_without_tag_from_G(G, args.input_path+'train_checkin_file.txt', args.input_path+'friendship_file.txt')
    node2id, id2node = indexing(node_type)

    '''2.形成index G的阶段'''
    u_p_edges, p_p_edges, u_u_edges = change_edges_node_id(u_p_edges, p_p_edges, u_u_edges, node2id)
    poi_longi_lanti = change_dict_key(poi_longi_lanti, node2id)
    history = change_history_node_id(history, node2id)
    node_type = change_dict_key(node_type,node2id)
    G = get_G_from_edges(u_p_edges, p_p_edges, u_u_edges, history, args.epsilon)
    G = add_geo_type_info(G, poi_longi_lanti, node_type)
    # G = update_weight_by_geo(G, kappa=args.kappa)

    # write_G_to_file(G, 'C:/Users/lenovo/Desktop/G.txt')


    '''3.生成features 和 adj 的tensor'''
    if args.cuda:
        features = torch.eye(G.number_of_nodes()).to(device='cuda')
    else:
        features = torch.eye(G.number_of_nodes()).to(device='cpu') # + torch.randn((G.number_of_nodes(), G.number_of_nodes())) * 0.1
    adj = nx.adjacency_matrix(G)
    adj = preprocess_adj(adj, normalization='AugNormAdj')   # FAMENormAdj



    '''4.定义GCN模型训练'''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda: # False
        torch.cuda.manual_seed(args.seed) # G.number_of_nodes() # TODO: TODO: 特定features的话需要更改维度
    # gcn_model = get_model('GCN',nfeat=args.input_dim, nclass=args.output_dim, degree=args.degree, nhid=args.hidden_dim, dropout=args.dropout,cuda=args.cuda)
    gcn_model = get_model('GCN',nfeat=G.number_of_nodes(), nclass=args.output_dim, degree=args.degree, nhid=args.hidden_dim, dropout=args.dropout,cuda=args.cuda)
    optimizer_gcn = optim.Adam( gcn_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    '''5.查找训练和测试数据的索引'''
    LSTM_test_records_input, LSTM_test_target_poi_list, LSTM_test_user_list = gen_test_data(args.input_path, node2id, args.delt, args.test_sample_num, args.seed)
    LSTM_train_records_input, LSTM_train_records_output = gen_train_data(args.input_path, node2id)

    data_train = My_dataset(LSTM_train_records_input, LSTM_train_records_output)
    data_loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    '''6.随机游走的数据'''
    walker = RWGraph(G)
    adj_weight_dict = find_positive_samples(G, args.window_size)
    # walks_list = walker.simulate_walks(args.num_walks, args.walk_length, schema=None, isweighted=args.isweighted) # TODO: 有136个节点是没有边的，是单独的
    # walks_list = [col for row in walks_list for col in row]
    # vocab_list = Counter(walks_list).most_common() # 每个元素是一个元组[(539,347), (457,333)...]
    # word_counts = np.array([count[1] for count in vocab_list], dtype=np.float32) #
    # word_freqs = word_counts / np.sum(word_counts)
    # word_freqs = word_freqs ** (3. / 4.)
    # vocab_list = [item[0] for item in vocab_list]
    print("finished random walk!")

    # write_pkl(adj_weight_dict, args.input_path+'log_file/adj_weight_dict.pkl')
    # write_pkl(vocab_list, args.input_path+'log_file/vocab_list.pkl')
    # write_pkl(word_freqs, args.input_path+'log_file/word_freqs.pkl')

    # adj_weight_dict = read_pkl(args.input_path+'log_file/adj_weight_dict.pkl')
    vocab_list = read_pkl(args.input_path+'log_file/vocab_list.pkl')
    word_freqs = read_pkl(args.input_path+'log_file/word_freqs.pkl')
    num_node = len(word_freqs)


    '''7.1加载训练数据训练-GCN'''
    loss_func_mse = nn.MSELoss()
    loss_func_cos = nn.CosineEmbeddingLoss(margin=0., reduction='none')
    train_loss = []
    min_valid_loss = np.inf
    print('Training...')

    for i in range(args.epochs):
        gcn_model.train()
        optimizer_gcn.zero_grad()
        output_embedding = gcn_model(features, adj)
        train_loss_rw = objective_rw(output_embedding, args.negk, adj_weight_dict, vocab_list, word_freqs)
        log_string = ('iter: [{:d}/{:d}], train_loss_rw: {:0.6f}, lr: {:0.7f}').format((i), args.epochs, train_loss_rw, optimizer_gcn.param_groups[0]['lr'])
        print(log_string)
        train_loss_rw.backward()
        optimizer_gcn.step()

        if i % 10 == 0:
            time_str = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
            write_pkl(output_embedding, args.input_path +'log_file/{}_output_embedding_epoch_{}_dim_{}.pkl'.format(time_str, i, args.output_dim))

            '''8.实验结果评价'''
            input_user_poi_list = []
            for user_records in LSTM_test_records_input:
                input_user_poi_list.append( user_records[-1] )

            accuracy, precision, recall, ndcg, hit_ratio, MAP = evaleate_no_lstm(args.input_path, output_embedding, input_user_poi_list, LSTM_test_target_poi_list, node_type, i)
    print('over!')