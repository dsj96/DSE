'''
Descripttion:
version:
Author: ShaojieDai
Date: 2021-05-04 21:03:05
LastEditors: sueRimn
LastEditTime: 2021-05-20 15:09:29
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', type=bool, default=False,
                        help='If use CUDA.')

    parser.add_argument('--input_path', type=str, default='dataset/toyset/',
                        help='Input dataset path') # 'dataset/toyset/'  'dataset/brightkite/' 'dataset/foursquare/'

    parser.add_argument('--model_opt', type=str, default='GCN', # SGC GCN
                        help='Set the model type') # model_opt test_sample_num

    parser.add_argument('--test_sample_num', type=int, default=1000,
                        help='Set the test number samples') # model_opt

    parser.add_argument('--epsilon', type=float, default=0.5,
                        help='Set the epsilon')

    parser.add_argument('--theta', type=float, default=24,
                        help='Set the theta')

    parser.add_argument('--kappa', type=int, default=-2, #
                        help='Set the kappa')

    parser.add_argument('--delt', type=int, default=6, #
                        help='Set the delt to predict')

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')

    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')

    parser.add_argument('--input_dim',  type=int, default=35, # 17161
                        help='Number of hidden units.')

    parser.add_argument('--hidden_dim',  type=int, default=128,
                        help='Number of hidden units.')

    parser.add_argument('--output_dim',  type=int, default=128,
                        help='Number of hidden units.')

    parser.add_argument('--num_layers',  type=int, default=2,
                        help='Number of lstm layers.')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')

    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')

    parser.add_argument('--negk', type=int, default=5,
                        help='find the topk most not similary item.')

    parser.add_argument('--num_walks', type=int, default=20,
                        help='.')

    parser.add_argument('--walk_length', type=float, default=10,
                        help='.')

    parser.add_argument('--isweighted', type=bool, default=True,
                        help='walk is weighted.')

    parser.add_argument('--max_seq_len', type=int, default=30,
                        help='walk is weighted.')

    parser.add_argument('--batch_size', type=int, default=3000,
                        help='.')

    parser.add_argument('--n_trials', type=int, default=100,
                        help='.')

    parser.add_argument('--window_size', type=int, default=3,
                        help='.')

    parser.add_argument('--min_seq_len', type=int, default=4,
                        help='min seqence len for train selected data!')


    args = parser.parse_args()

    return args