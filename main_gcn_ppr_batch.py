'''
Descripttion: 
version: 
Author: ShaojieDai
Date: 2021-05-08 23:06:23
LastEditors: Please set LastEditors
LastEditTime: 2021-09-14 19:44:44
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


import optuna

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
from extract_feature import extract_forsquare_features

def preprocess_foursquare_features(poi_feature_dict, user_feature_dict, node2id):
    for k,v in user_feature_dict.items():
        num_feature = len(v)
        break

    init_features = torch.zeros((len(node2id), num_feature))
    for node, id_ in node2id.items():
        if node in poi_feature_dict.keys():
            init_features[id_] = torch.FloatTensor(poi_feature_dict[node])
        if node in user_feature_dict.keys():
            init_features[id_] = torch.FloatTensor(user_feature_dict[node])
    return init_features

def preprocess_train_data(LSTM_train_records_input, LSTM_train_records_output, max_seq_len):
    '''
    @name: ShaojieDai
    @Date: 2021-05-18 23:14:21
    @msg: 将过长的序列拆分成短的
    @param {*}
    @return {*}
    '''
    new_LSTM_train_records_input = []
    for user_history in LSTM_train_records_input:
        len_history = len(user_history)
        if len_history <= max_seq_len:
            new_LSTM_train_records_input.append(user_history)
        else:
            new_LSTM_train_records_input.append(user_history[:max_seq_len])

    new_LSTM_train_records_output = []
    for user_history in LSTM_train_records_output:
        len_history = len(user_history)
        if len_history <= max_seq_len:
            new_LSTM_train_records_output.append(user_history)
        else:
            new_LSTM_train_records_output.append(user_history[:max_seq_len])
    return new_LSTM_train_records_input, new_LSTM_train_records_output

def preprocess_test_data(LSTM_test_records_input, max_seq_len):

    new_LSTM_test_records_input = []
    for user_history in LSTM_test_records_input:
        len_history = len(user_history)
        if len_history <= max_seq_len:
            new_LSTM_test_records_input.append(user_history)
        else:
            new_LSTM_test_records_input.append(user_history[:max_seq_len])

    return new_LSTM_test_records_input


def objective(trial):
    config = {
        "hy_RW"     : trial.suggest_loguniform('hy_RW' , 1e-5, 1e5),
        # "hy_RW"     : trial.suggest_uniform('hy_RW' , 1e-2, 100),
        "hy_seq"    : trial.suggest_loguniform('hy_seq' , 1e-5, 1e5),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    '''1.都是过程文件，暂时没有到形成index G的阶段'''
    '''加上check_set的目的主要是为了检查user_set和poi_set是否有交集，即poi和user是否有相同id的情况 interset_set原来=13'''
    user_set, poi_set = load_training_data_check_set(args.input_path+'train_checkin_file.txt', args.input_path+'friendship_file.txt', istag=False, isweek=False, hour_interval=4, edge_interval = args.theta)

    u_p_edges, p_p_edges, u_u_edges = load_training_data(args.input_path+'train_checkin_file.txt', args.input_path+'friendship_file.txt', istag=False, isweek=False, hour_interval=4) # , edge_interval = args.theta, interset_set=user_set&poi_set
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


    '''3.生成features 和 adj 的tensor'''
    if args.input_path == 'dataset/foursquare/':
        poi_feature_dict, user_feature_dict = extract_forsquare_features()
        if args.cuda:
            features = preprocess_foursquare_features(poi_feature_dict, user_feature_dict, node2id).to(device='cuda')
        else:
            features = preprocess_foursquare_features(poi_feature_dict, user_feature_dict, node2id)
    else:
        if args.cuda:
            features = torch.eye(G.number_of_nodes()).to(device='cuda')
        else:
            features = torch.eye(G.number_of_nodes()).to(device='cpu') # + torch.randn((G.number_of_nodes(), G.number_of_nodes())) * 0.1

    adj = nx.adjacency_matrix(G)
    if args.cuda:
        adj = preprocess_adj(adj, normalization='AugNormAdj').to(device='cuda')   # FAMENormAdj
    else:
        adj = preprocess_adj(adj, normalization='AugNormAdj')


    '''4.定义模型训练'''
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda: # False
        torch.cuda.manual_seed(args.seed)
    combine_model = COMBINE_MODEL(args.model_opt, features.shape[1], args.output_dim, args.degree, args.hidden_dim, args.num_layers, args.dropout, args.cuda)
    optimizer = optim.Adam( itertools.chain(combine_model.gcn_model.parameters(), combine_model.seq_model.parameters()) , lr=args.lr, weight_decay=args.weight_decay)


    '''5.查找训练和测试数据的索引'''
    LSTM_test_records_input, LSTM_test_target_poi_list, LSTM_test_user_list = gen_test_data(args.input_path, node2id, args.delt, args.test_sample_num, args.seed)
    LSTM_train_records_input, LSTM_train_records_output = gen_train_data(args.input_path, node2id, args.theta, args.min_seq_len) # file_path, node2id, delt, min_seq_len

    LSTM_train_records_input, LSTM_train_records_output = preprocess_train_data(LSTM_train_records_input, LSTM_train_records_output, args.max_seq_len)
    LSTM_test_records_input = preprocess_test_data(LSTM_test_records_input, args.max_seq_len)

    data_train = My_dataset(LSTM_train_records_input, LSTM_train_records_output)
    data_loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)


    '''6.随机游走'''
    # walker = RWGraph(G)
    # walks_list = walker.simulate_walks(args.num_walks, args.walk_length, schema=None, isweighted=args.isweighted) # TODO: 有136个节点是没有边的，是单独的
    # walks_list = [col for row in walks_list for col in row]
    # vocab_list = Counter(walks_list).most_common() # 每个元素是一个元组[(539,347), (457,333)...]
    # word_counts = np.array([count[1] for count in vocab_list], dtype=np.float32) #
    # word_freqs = word_counts / np.sum(word_counts)
    # word_freqs = word_freqs ** (3. / 4.)
    adj_weight_dict = find_positive_samples(G, args.window_size)
    # vocab_list = [item[0] for item in vocab_list]
    # print("finished random walk!")


    # adj_weight_dict = read_pkl(args.input_path + 'log_file/adj_weight_dict.pkl')
    vocab_list = read_pkl(args.input_path + 'log_file/vocab_list.pkl')
    word_freqs = read_pkl(args.input_path + 'log_file/word_freqs.pkl')
    num_node = len(word_freqs)


    '''7.加载训练数据训练'''
    loss_func_mse = nn.MSELoss()
    loss_func_cos = nn.CosineEmbeddingLoss(margin=0., reduction='none')
    train_loss = []
    min_valid_loss = np.inf
    print('Training...')
    # torch.autograd.set_detect_anomaly(True)
    for i in range(args.epochs):
        combine_model.train()
        # output_embedding = row_normalize(output_embedding)
        '''batch_size开始'''
        for src, trg in tqdm(data_loader_train):
            optimizer.zero_grad()
            prediction_list, output_embedding = combine_model(features, adj, src) # prediction_list 每一个元素是一个tensor (X,1,emd_dim)的形状


            seq_loss = 0.
            for idx,poi_list in enumerate(trg): # LSTM_train_records_output每一个元素是一个list [] len=X
                # seq_loss = loss_func_mse(output_embedding[poi_list], prediction_list[idx]) + seq_loss        # TODO:

                if args.cuda:
                    target = -torch.ones(len(prediction_list[idx])).to('cuda')
                else:
                    target = -torch.ones(len(prediction_list[idx]))

                seq_loss = torch.sum(loss_func_cos(output_embedding[poi_list], prediction_list[idx], target=target)) + seq_loss
            train_loss_rw = objective_rw(output_embedding, args.negk, adj_weight_dict, vocab_list, word_freqs)
            loss = config["hy_RW"]*train_loss_rw + config["hy_seq"]*seq_loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss)

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, lr: {:0.7f}').format((i), args.epochs, train_loss[-1], optimizer.param_groups[0]['lr'])
        print(log_string)
        log(args.input_path + 'log_file/LSTM.log', log_string)

        if i == args.epochs-1 or  i == 0: # 在偶数次的循环中打印test输出
            combine_model.eval()
            time_str = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
            write_pkl(output_embedding, args.input_path +'log_file/{}_mearge_output_embedding_epoch_{}_dim_{}.pkl'.format(time_str, i, args.output_dim))
            prediction_list, output_embedding = combine_model(features, adj, LSTM_test_records_input)
            candidate_pois_tensor = []
            for step, prediction in enumerate(prediction_list): # list的第一个元素(1039,256) (67,256)
                prediction = prediction[-1]
                candidate_pois_tensor.append(prediction)
            # time_str = time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))
            # write_pkl(output_embedding, args.input_path +'log_file/{}_mearge_output_embedding_epoch_{}_dim_{}.pkl'.format(time_str, i, args.output_dim))
            accuracy, precision, recall, ndcg, hit_ratio, MAP = evaleate(args.input_path, candidate_pois_tensor, LSTM_test_target_poi_list, node_type, output_embedding)
    return -accuracy[1]

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

        # cur_loss_term =  - torch.sum(F.logsigmoid(torch.pairwise_distance( positive_embedding, cur_embedding, p=2) )) \
        #                  - torch.sum(torch.log( 1. - torch.sigmoid(torch.pairwise_distance(negative_embedding, cur_embedding, p=2 ) ) ))

        loss_term = loss_term + cur_loss_term
    return loss_term



'''1.自己编写的'''
class My_dataset(Dataset):
    def __init__(self, LSTM_train_records_input, LSTM_train_records_output):
        '''max_seq_len=-10 negative LSTM_train_records_input= 3 layer list'''
        super().__init__()
        self.LSTM_train_records_input   = LSTM_train_records_input
        self.LSTM_train_records_output  = LSTM_train_records_output

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



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args() # 'dataset/toyset/'

    '''0.optuna 相关参数'''
    study_name = args.input_path.split('/')[-2]

    study = optuna.create_study(study_name=study_name, storage='sqlite:///{}log_file/{}.db'.format(args.input_path, study_name), load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials)
    print('over!')

    '''1.读取db文件，查看最佳参数'''
    study = optuna.create_study(study_name=study_name, storage='sqlite:///{}log_file/{}.db'.format(args.input_path, study_name), load_if_exists=True)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)
    df.to_csv('{}log_file/{}_{}.csv'.format( args.input_path, time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time()))), study_name)

    '''2.可视化优化记录 函数都是返回：A plotly.graph_objs.Figure object. '''
    # import plotly as py
    # # py.offline.plot(figure,auto_open=True)
    # study_name = args.input_path.split('/')[-2]
    # study = optuna.create_study(study_name=study_name, storage='sqlite:///{}log_file/{}.db'.format(args.input_path, study_name), load_if_exists=True)
    # # # 将参数关系画成等高线图。
    # figure1 = optuna.visualization.plot_contour(study, params=['hy_RW', 'hy_volume_current', 'hy_volume_recent'])
    # py.offline.plot(figure1,auto_open=True)
    # # # 画出一个 study 中的全部 trial 的中间值。
    # figure2 = optuna.visualization.plot_intermediate_values(study) # 无效
    # py.offline.plot(figure2,auto_open=True)
    # # # 画出一个 study 中所有 trial 的优化历史记录。
    # figure3 = optuna.visualization.plot_optimization_history(study)
    # py.offline.plot(figure3,auto_open=True)
    # # # 绘制一个 study 中高维度参数的关系图。
    # figure4 = optuna.visualization.plot_parallel_coordinate(study, params=['hy_RW', 'hy_volume_current', 'hy_volume_recent'])
    # py.offline.plot(figure3,auto_open=True)
    # # # 画出超参数的重要性
    # figure5 = optuna.visualization.plot_param_importances(study, params=['hy_RW', 'hy_volume_current', 'hy_volume_recent'])
    # py.offline.plot(figure5,auto_open=True)
    # # # 绘制一个 study 中的参数关系切片图。
    # figure6 = optuna.visualization.plot_slice(study, params=['hy_RW', 'hy_volume_current', 'hy_volume_recent'])
    # py.offline.plot(figure6,auto_open=True)
    # '''显示'''
    # py.offline.plot(fig,auto_open=True) # filename="iris1.html"
    # print('over!')