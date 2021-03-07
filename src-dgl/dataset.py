# -*- coding: utf-8 -*-
"""
Created on 31/3/2019
@author: RuihongQiu
"""

import pickle
import torch
from dgl.data import DGLDataset
import dgl

class MultiSessionsGraph(DGLDataset):
    """Every session is a graph."""
    def __init__(self, name, raw_dir):
        """
        Args:
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
            phrase: 'train' or 'test'
        """
        assert name in ['train.txt', 'valid.txt', 'test.txt']
        super(MultiSessionsGraph, self).__init__(name=name, raw_dir=raw_dir)

    def process(self):
        data = pickle.load(open(self.raw_dir + '/' + self.name, 'rb'))
        data_list = []
        
        for sequences, y in zip(data[0], data[1]):
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []
            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            del senders[-1]    # the last item is a receiver
            del receivers[0]    # the first item is a sender
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)

            self.graph = dgl.graph((senders, receivers))
            self.graph.ndata['x'] = x
            self.graph.ndata['label'] = y
            self.graph.edata['weight'] = edge_features

            data_list.append(DGLGraph(x=x, edge_index=edge_index, y=y))
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_classes(self):
        """Number of classes."""
        return 2

    @property
    def data(self):
        deprecate_property('dataset.data', 'dataset[0]')
        return self._data

    def __getitem__(self, idx):
        r""" Get graph object

        Parameters
        ----------
        idx : int
            Item index, KarateClubDataset has only one graph object

        Returns
        -------
        :class:`dgl.DGLGraph`

            graph structure and labels.

            - ``ndata['label']``: ground truth labels
        """
        assert idx == 0, "This dataset has only one graph"
        return self._graph

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1