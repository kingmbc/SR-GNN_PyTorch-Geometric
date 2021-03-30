# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""

import os
import argparse
import logging
from tqdm import tqdm
from dataset import MultiSessionsGraph
from model import *
from train import forward
from tensorboardX import SummaryWriter
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--top_k', type=int, default=20, help='top K indicator for evaluation')
opt = parser.parse_args()
logging.warning(opt)



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cur_dir = os.getcwd()
    train_filename, test_filename = 'train.txt', 'test.txt'
    dataset_name = opt.dataset
    if 'yoochoose' in opt.dataset:
        dataset_name = 'yoochoose'
        train_filename = opt.dataset + '-' + train_filename
        test_filename = opt.dataset + '-' + test_filename

    train_dataset = MultiSessionsGraph(name=train_filename,
                                       raw_dir=cur_dir + '/../../../_data/' + dataset_name + '/processed/',
                                       save_dir=cur_dir + '/../../../_data/' + dataset_name + '/saved/',
                                       force_reload=True)
    num_train = len(train_dataset)
    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    train_loader = GraphDataLoader(train_dataset, batch_size=opt.batch_size,
                                   # sampler=train_sampler,
                                   shuffle=True, drop_last=False)

    test_dataset = MultiSessionsGraph(name=test_filename,
                                      raw_dir=cur_dir + '/../../../_data/' + dataset_name + '/processed/',
                                      save_dir=cur_dir + '/../../../_data/' + dataset_name + '/saved/',
                                      force_reload=True)
    num_test = len(test_dataset)
    test_sampler = SubsetRandomSampler(torch.arange(num_test))
    test_loader = GraphDataLoader(test_dataset, batch_size=opt.batch_size,
                                  # sampler=test_sampler,
                                  shuffle=False, drop_last=False)

    log_dir = os.path.join(cur_dir, 'log', str(opt.dataset), str(opt))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)

    if 'diginetica' in opt.dataset:
        n_node = 43097
    elif 'yoochoose' in opt.dataset:
        n_node = 37483
    else:
        n_node = 309

    model = GNNModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    logging.warning(model)
    
    for epoch in tqdm(range(opt.epoch)):
        forward(model, train_loader, device, writer, epoch, top_k=opt.top_k, optimizer=optimizer, train_flag=True)
        with torch.no_grad():
            forward(model, test_loader, device, writer, epoch, top_k=opt.top_k, train_flag=False)
        scheduler.step()


if __name__ == '__main__':
    main()
