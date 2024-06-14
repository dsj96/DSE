'''
Descripttion: 
version: 
Author: ShaojieDai
Date: 2021-05-08 09:42:51
LastEditors: sueRimn
LastEditTime: 2021-05-19 13:32:47
'''
'''
Descripttion: 
version: 
Author: ShaojieDai
Date: 2021-05-03 16:56:35
LastEditors: sueRimn
LastEditTime: 2021-05-07 11:08:53
'''

import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math

from tqdm import tqdm


class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass, degree, cuda):
        super(SGC, self).__init__()
        self.degree = degree
        self.W = nn.Linear(nfeat, nclass, bias=True) # θ SGC论文公式8的上面段落
        # self.alpha = Variable(torch.FloatTensor([1. for i in range(self.degree)]), requires_grad=True)
        self.alpha = nn.Parameter(Variable(torch.FloatTensor([1 for i in range(self.degree)]), requires_grad=True))

    def init(self):
        stdv = 1. / math.sqrt(self.alpha.weight.size(0))
        self.alpha.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, weight_adj_list):
        # return F.relu(self.W(x))
        # return F.sigmoid(self.W(x))
        # return F.tanh(self.W(x))
        if type(weight_adj_list) == list:
            weight_aug_adj = 0
            for idx,item in enumerate(weight_adj_list):
                weight_aug_adj = weight_aug_adj + self.alpha[idx] * item
            return self.W(torch.spmm(weight_aug_adj, x))
        else:
            # TODO: 在sgc_precompute中已经计算了features,所以不是 return self.W(torch.spmm(weight_aug_adj, x))
            return self.W(weight_adj_list)

class GraphConvolution(Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)

        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv*10, stdv*10)

    def forward(self, input, adj):
        support = self.W(input) # XW
        output = torch.spmm(adj, support) # AXW
        return output

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout, degree):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.degree = degree

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
            # x = F.sigmoid(x)
            # x = torch.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class STNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, cuda): # 128   32  2
        super(STNN, self).__init__()
        self.lstm_model = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            dropout     = dropout,
            batch_first = False         # (seq, batch, feature)
        )
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_out = nn.Linear(hidden_dim, self.output_dim) # (32,64)
        self.cuda       = cuda
        self.h_s = None
        self.h_c = None
        self.init()

    def init(self):
        # stdv = 1. / math.sqrt(self.hidden_out.weight.size(1)) # args.hidden_size
        # self.hidden_out.weight.data.uniform_(-stdv, stdv)
        # self.hidden_out.bias.data.uniform_(-stdv, stdv)

        # self.lstm_model.bias_hh_l0.data.uniform_(-stdv, stdv)
        # self.lstm_model.bias_hh_l1.data.uniform_(-stdv, stdv)
        # self.lstm_model.bias_ih_l0.data.uniform_(-stdv, stdv)
        # self.lstm_model.bias_ih_l1.data.uniform_(-stdv, stdv)

        # self.lstm_model.weight_hh_l0.data.uniform_(-stdv, stdv)
        # self.lstm_model.weight_hh_l1.data.uniform_(-stdv, stdv)
        # self.lstm_model.weight_ih_l0.data.uniform_(-stdv, stdv)
        # self.lstm_model.weight_ih_l1.data.uniform_(-stdv, stdv)
        if self.cuda:
            self.lstm_model.to('cuda')
            self.hidden_out.to('cuda')


    def forward(self, x):
        lstm_out, (h_s, h_c) = self.lstm_model(x)
        output = self.hidden_out(lstm_out)
        return output

class COMBINE_MODEL(nn.Module):
    def __init__(self, model_opt, input_dim, output_dim, degree, hidden_dim, num_layers, dropout, cuda ):
        super(COMBINE_MODEL, self).__init__()
        self.model_opt  = model_opt
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.degree     = degree
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout
        self.cuda       = cuda

        self.loss_func = nn.MSELoss()
        # self.gcn_model = gcn_model get_model(model_opt=args.model_opt, nfeat=args.input_dim, nclass=args.output_dim, degree=args.degree, nhid=args.hidden_dim, dropout=args.dropout, cuda=args.cuda)
        # self.seq_model = seq_model STNN(input_dim=args.output_dim*2, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout)
        self.gcn_model =  get_model(model_opt=self.model_opt, nfeat=self.input_dim, nclass=self.output_dim, degree=self.degree, nhid=self.hidden_dim, dropout=self.dropout, cuda=self.cuda)
        self.seq_model =  STNN(input_dim=self.output_dim*2, hidden_dim=self.hidden_dim, output_dim=self.output_dim, num_layers=self.num_layers, dropout=self.dropout, cuda=self.cuda) # (64*2,32,2)

    def forward(self, features, adj, LSTM_train_records_input): # LSTM_train_records_input= [ [[],[],[]...],   ]
        output_embedding = self.gcn_model(features, adj)
        prediction_list = []
        for user_records in LSTM_train_records_input:
            cur_input_list = []
            for user_poi in user_records:                    # TODO:
                cur_input_list.append(output_embedding[user_poi].reshape(1, -1))        # (2,64)->(1,128)
            cur_seq_input_data = torch.cat(cur_input_list, dim=0).unsqueeze(1)    # (7,128)->(7,1,128)
                          # (7,1,64)
            prediction = self.seq_model(cur_seq_input_data).squeeze(1)                 # (7,1,64)-(7,64)
            prediction_list.append(prediction)

        return prediction_list, output_embedding


def get_model(model_opt, nfeat, nclass, degree, nhid, dropout, cuda): # ('SGC', 1433, 7, 0, 0, False)
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout,
                    degree=degree)
    elif model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass,
                    degree=degree,
                    cuda=cuda) # (1433, 7)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model


if __name__ == '__main__':
    seq_model = STNN(input_dim=16,
                    hidden_dim=16,
                    num_layers=2,
                    dropout=0.1)
    print("over")
