# -*- coding: utf-8 -*-
"""
Created on 5/4/2019
@author: RuihongQiu
"""


import numpy as np
from tqdm import tqdm

ZERO_PADDING_COUNT = 1

def forward(model, loader, device, writer, epoch, top_k=20, optimizer=None, train_flag=True):
    if train_flag:
        model.train()
    else:
        model.eval()
        hit, mrr = [], []

    mean_loss = 0.0
    updates_per_epoch = len(loader)

    for i, (batch_graph, batch_target) in enumerate(tqdm(loader)):
        if train_flag:
            optimizer.zero_grad()
        batch_graph = batch_graph.to(device)
        batch_target = batch_target.to(device)
        if batch_graph.number_of_edges() == 0:
            continue
        scores = model(batch_graph, batch_graph.ndata['x'] - ZERO_PADDING_COUNT)
        targets = batch_target - ZERO_PADDING_COUNT
        loss = model.loss_function(scores, targets)

        if train_flag:
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)
        else:
            sub_scores = scores.topk(top_k)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))

        mean_loss += loss / batch_graph.batch_size

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
        print('loss/train_loss', mean_loss.item(), epoch)

    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        writer.add_scalar('index/hit', hit, epoch)
        writer.add_scalar('index/mrr', mrr, epoch)
        print('index/hit', hit, epoch)
        print('index/mrr', mrr, epoch)