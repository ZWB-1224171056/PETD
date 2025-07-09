# -*- coding:utf-8 -*-
# @Time  : 2021/3/10 10:28
# @Author: ZWB
# @File  : utils.py
import argparse
import numpy as np


# 参数加载
def parse_args(script='None'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', type=str, default='CIFAR_FS')
    parser.add_argument('--dataset_test', type=str, default='CIFAR_FS')
    parser.add_argument('--num_test', type=str, default="Best")#Best
    parser.add_argument('--task_num', type=int, default=256)
    parser.add_argument('--backbone', type=str, default='Res12')    # conv4,Res12
    parser.add_argument('--batch_size', type=int, default=64, metavar='BatchSize')
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--n_query', type=int, default=5)
    parser.add_argument('--cluster_num', type=int, default=512, metavar='cluster_num')
    parser.add_argument('--n_test_way', type=int, default=5, metavar='NTESTWAY')
    parser.add_argument('--n_test_shot', type=int, default=1, metavar='NTESTSHOT')
    parser.add_argument('--n_test_query', type=int, default=15, metavar='NTESTQUERY')
    parser.add_argument('--n_epochs', type=int, default=400, metavar='NEPOCHS')
    parser.add_argument('--n_prob_epochs', type=int, default=400, metavar='NEPOCHS')
    parser.add_argument('--exp', type=str, default='1') 
    parser.add_argument('--n_test_episodes', type=int, default=10000, metavar='NTESTEPI')
    parser.add_argument('--n_eval_episodes', type=int, default=1000)
    parser.add_argument('--con', type=int, default=0, metavar='CON',
                        help="iteration to restore params")
    parser.add_argument('--augment', type=str, default='AMDIM')#SimCLR，AMDIM
    parser.add_argument('--similarity',type=str,default='sns')#cosine，sns
    # parse gpu
    parser.add_argument('--device',default='cuda')
    # train params
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--multi_gpu', action='store_true', default=True)
    parser.add_argument('--unsupervised', default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--train_size', type=int,default=84)
    parser.add_argument('--test_size', type=int,default=84)
    parser.add_argument('--m', type=int,default=0)
    parser.add_argument('--pmax', type=float,default=1.0)
    parser.add_argument('--dual_lr', type=float, default=4, help='dual learning rate')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of lower-bound')
    parser.add_argument('--train_update_steps', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='../data')

    return parser.parse_args()

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text, end='')
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
