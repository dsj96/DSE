'''
Descripttion:
version:
Author: ShaojieDai
Date: 2021-05-04 21:04:37
LastEditors: sueRimn
LastEditTime: 2021-05-12 15:40:01
'''
# torch
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
        while( len(set(negative_list) - set(cur_adj.keys())) < negk ): # TODO: 可能出现死循环，几率很小
            negative_list = random.choices(population=vocab_list, weights=word_freqs, k=negk)
        positive_embedding = output[list(cur_adj.keys())] # pos_num*dim
        weight_positive_tensor = torch.tensor([item["weight"] for item in list(cur_adj.values())],dtype=torch.float32).unsqueeze(0) # (1, pos_num)

        negative_embedding = output[negative_list] # neg_num*dim
        # cur_loss_term =  - sum(F.logsigmoid( positive_embedding.mm(cur_embedding.unsqueeze(1)) )) - sum(torch.log( 1. - torch.sigmoid(negative_embedding.mm(cur_embedding.unsqueeze(1))) ))
        # cur_loss_term =  - sum(F.logsigmoid(weight_positive_tensor.mm( positive_embedding.mm(cur_embedding.unsqueeze(1)) ) )) - sum(F.logsigmoid( 1. - negative_embedding.mm(cur_embedding.unsqueeze(1)) ))
        # 下面这个带权重wij                          (1,8)                                             (8,128)             (1,128)
        # cur_loss_term =  - sum(F.logsigmoid(weight_positive_tensor.mm( torch.cosine_similarity( positive_embedding, cur_embedding.unsqueeze(0), dim=1).unsqueeze(1) ))) \
        #                  - sum(torch.log( 1. - torch.sigmoid(torch.cosine_similarity(negative_embedding, cur_embedding.unsqueeze(0), dim=1 ) ) ))
        # 下面这个不带权重wij
        cur_loss_term =  - torch.sum(F.logsigmoid(torch.cosine_similarity( positive_embedding, cur_embedding, dim=-1) )) \
                         - torch.sum(torch.log( 1. - torch.sigmoid(torch.cosine_similarity(negative_embedding, cur_embedding, dim=-1 ) ) ))

        # cur_loss_term_test =  - sum(sum(sum(F.logsigmoid(torch.cosine_similarity( positive_embedding, cur_embedding.unsqueeze(1), dim=-1).unsqueeze(1) )))) \
        #                  - sum(sum(torch.log( 1. - torch.sigmoid(torch.cosine_similarity(negative_embedding, cur_embedding.unsqueeze(1), dim=-1 ) ) )))

        loss_term = loss_term + cur_loss_term
    return loss_term

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

    # write_G_to_file(G, 'C:/Users/ShaojieDai/Desktop/G.txt')

    '''3.生成features 和 adj 的tensor'''
    if args.cuda:
        features = torch.eye(G.number_of_nodes()).to(device='cuda')
    else:
        features = torch.eye(G.number_of_nodes()).to(device='cpu')
    adj = nx.adjacency_matrix(G)
    adj = preprocess_adj(adj, normalization='AugNormAdj')   # FAMENormAdj


    '''4.定义模型训练'''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda: # False
        torch.cuda.manual_seed(args.seed)
    combine_model = COMBINE_MODEL(args.model_opt, args.input_dim, args.output_dim, args.degree, args.hidden_dim, args.num_layers, args.dropout, args.cuda, args.max_seq_len)
    optimizer = optim.Adam( itertools.chain(combine_model.gcn_model.parameters(), combine_model.seq_model.parameters()) , lr=args.lr, weight_decay=args.weight_decay)


    '''5.查找训练和测试数据的索引'''
    LSTM_test_records_input, LSTM_test_target_poi_list, LSTM_test_user_list = gen_test_data(args.input_path, node2id, args.delt, args.test_sample_num, args.seed)
    LSTM_train_records_input, LSTM_train_records_output = gen_train_data(args.input_path, node2id)
    # train_pairs = zip(LSTM_train_records_input, LSTM_train_records_output)
    # LSTM_train_records_input = LSTM_train_records_input[:100]
    # LSTM_train_records_output = LSTM_train_records_output[:100]


    '''6.计算随机游走目标函数需要的相关数据'''
    # walker = RWGraph(G)
    # walks_list = walker.simulate_walks(args.num_walks, args.walk_length, schema=None, isweighted=args.isweighted) # TODO: 有136个节点是没有边的，是单独的

    # walks_list = [col for row in walks_list for col in row]
    # vocab_list = Counter(walks_list).most_common() # 每个元素是一个元组[(539,347), (457,333)...]

    # word_counts = np.array([count[1] for count in vocab_list], dtype=np.float32) #
    # word_freqs = word_counts / np.sum(word_counts)
    # word_freqs = word_freqs ** (3. / 4.)
    # adj_weight_dict = find_positive_samples(G)
    # vocab_list = [item[0] for item in vocab_list]
    # print("finished random walk!")
    adj_weight_dict = read_pkl(args.input_path + 'log_file/adj_weight_dict.pkl')
    vocab_list = read_pkl(args.input_path + 'log_file/vocab_list.pkl')
    word_freqs = read_pkl(args.input_path + 'log_file/word_freqs.pkl')
    num_node = len(word_freqs)

    '''7.加载训练数据训练'''
    loss_func = nn.MSELoss()
    train_loss = []
    min_valid_loss = np.inf
    print('Training...')
    # torch.autograd.set_detect_anomaly(True)
    for i in range(args.epochs):
        combine_model.train()
        optimizer.zero_grad()

        prediction_list, output_embedding = combine_model(features, adj, LSTM_train_records_input) # prediction_list 每一个元素是一个tensor (X,1,emd_dim)的形状
        seq_loss = 0.
        for idx,poi_list in enumerate(LSTM_train_records_output): # LSTM_train_records_output每一个元素是一个list [] len=X
            seq_loss = loss_func(output_embedding[poi_list[args.max_seq_len:]], prediction_list[idx]) + seq_loss        # TODO:
        train_loss_rw = objective_rw(output_embedding, args.negk, adj_weight_dict, vocab_list, word_freqs)
        loss = train_loss_rw + seq_loss*num_node/10.
        loss.backward()
        optimizer.step()
        train_loss.append(loss)


        # random.shuffle(train_pairs)
        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, lr: {:0.7f}').format((i + 1), args.epochs, train_loss[-1], optimizer.param_groups[0]['lr'])
        print(log_string)
        log(args.input_path + 'log_file/LSTM.log', log_string)


        if i % 10 == 0: # 在偶数次的循环中打印test输出
            combine_model.eval()
            prediction_list, output_embedding = combine_model(features, adj, LSTM_test_records_input)
            candidate_pois_tensor = []
            for step, prediction in enumerate(prediction_list): # list的第一个元素(1039,256) (67,256)
                prediction = prediction[-1]
                candidate_pois_tensor.append(prediction)

            accuracy, precision, recall, ndcg, hit_ratio, MAP = evaleate(args.input_path, candidate_pois_tensor, LSTM_test_target_poi_list, node_type, output_embedding)

    print('over!')