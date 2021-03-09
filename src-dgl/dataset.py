# -*- coding: utf-8 -*-

import os
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
import pickle
import torch
from dgl.data import DGLDataset
import dgl
from tqdm.contrib import tzip


class MultiSessionsGraph(DGLDataset):
    """Every session is a graph."""
    def __init__(self, name, raw_dir=None, force_reload=False, save_dir=None):
        """
        Args:
            name: 'train.txt' or 'test.txt'
            root: 'sample', 'yoochoose1_4', 'yoochoose1_64' or 'diginetica'
        """
        super(MultiSessionsGraph, self).__init__(name=name, raw_dir=raw_dir, save_dir=save_dir, force_reload=force_reload)

    def process(self):
        data = pickle.load(open(os.path.join(self.raw_dir, self.name), 'rb'))
        self.graphs = []
        self.labels = []
        self.all_seqs = []
        self.max_seq_length = 0
        self.max_node_id = 0
        self.max_num_unique_node = 0

        for sequences, y in tzip(data[0], data[1]):
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            unique_nodes = []
            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    unique_nodes.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]

            del senders[-1]  # the last item is a receiver
            del receivers[0]  # the first item is a sender
            g = dgl.graph((senders, receivers), num_nodes=len(unique_nodes))
            g.ndata['x'] = torch.tensor(unique_nodes, dtype=torch.long)
            g.edata['w'] = torch.ones(g.num_edges(), dtype=torch.float)

            # print(f"\n{g.nodes()}, {g.edges()}, {g.ndata['x'].squeeze()}")

            self.graphs.append(g)
            self.all_seqs.append(sequences)
            self.labels.append(y)

            if max(sequences) > self.max_node_id:
                self.max_node_id = max(sequences)

            if len(unique_nodes) > self.max_num_unique_node:
                self.max_num_unique_node = len(unique_nodes)

            if len(sequences) > self.max_seq_length:
                self.max_seq_length = len(sequences)

        # Convert the label list to tensor for saving.
        self.num_graphs = len(self.graphs)
        self.num_labels = len(self.labels)
        self.max_labels = max(self.labels)
        self.labels = torch.LongTensor(self.labels)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_{}_{}.bin'.format(self.name, self.hash))
        info_path = os.path.join(self.save_path, 'dgl_graph_{}_{}.pkl'.format(self.name, self.hash))
        label_dict = {'labels': self.labels}
        info_dict = {'num_graphs': self.num_graphs,
                     'num_labels': self.num_labels,
                     'max_labels': self.max_labels,
                     'max_node_id': self.max_node_id,
                     'max_num_unique_node': self.max_num_unique_node,
                     'max_seq_length': self.max_seq_length,
                     }
        save_graphs(str(graph_path), self.graphs, label_dict)
        save_info(str(info_path), info_dict)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_{}_{}.bin'.format(self.name, self.hash))
        info_path = os.path.join(self.save_path, 'dgl_graph_{}_{}.pkl'.format(self.name, self.hash))
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graphs = graphs
        self.labels = label_dict['labels']
        self.num_graphs = info_dict['num_graphs']
        self.num_labels = info_dict['num_labels']
        self.max_labels = info_dict['max_labels']
        self.max_node_id = info_dict['max_node_id']
        self.max_num_unique_node = info_dict['max_num_unique_node']
        self.max_seq_length = info_dict['max_seq_length']

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph_{}_{}.bin'.format(self.name, self.hash))
        info_path = os.path.join(self.save_path, 'dgl_graph_{}_{}.pkl'.format(self.name, self.hash))
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    def statistics(self):
        return self.graphs[0].ndata['x'].shape[1],\
            self.num_labels,\
            self.max_num_unique_node

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
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return len(self.graphs)