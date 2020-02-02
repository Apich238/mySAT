from generator import read_dataset, make_graph_2partial, make_graph_tree
import numpy as np

import random


class dataset:
    def __init__(self, data_file, graph_format, var_limit=6):
        self.data, self.labels = read_dataset(data_file)

        self.cdata = {}
        for i, d in enumerate(self.data):
            if graph_format == 'tree':
                graph_sz = 2 + len(d)
                al = set(np.concatenate(d).tolist())
                neg = len([x for x in al if x < 0])
                graph_sz += 2 * neg
                vs = set([abs(x) for x in al])
                graph_sz += len(vs)
            else:
                graph_sz = len(d)
                al = set([abs(x) for x in np.concatenate(d)])
                graph_sz += len(al)
            if graph_sz in self.cdata:
                self.cdata[graph_sz].append(i)
            else:
                self.cdata[graph_sz] = [i]

        self.var_limit = var_limit
        if graph_format == 'tree':
            self.mx_generator = make_graph_tree
        else:
            self.mx_generator = make_graph_2partial
        self.labels = np.asarray(self.labels)
        self.diff_var_annotations = var_limit > 1
        self.all_nodes = sorted(self.detect_all_nodes())

    def detect_all_nodes(self):
        an = []
        if self.mx_generator is make_graph_2partial:
            an = ['c']
        else:
            an = ['c', 'g', 'd', 'n', 'p']

        if self.diff_var_annotations:
            an.extend([f'v{i}' for i in range(self.var_limit)])
        else:
            an.extend(['v'])
        return an

    @staticmethod
    def normalize_adj_matrix_by_Kipf(a):
        D2 = np.diag(np.sqrt(np.sum(np.abs(a), axis=0))) ** -1
        D2[np.isinf(D2)] = 0.
        m = np.matmul(np.matmul(D2, a), D2) + np.eye(D2.shape[0])
        return m

    def int2onehot(self, l, all_labels):
        res = np.zeros(shape=[len(all_labels)], dtype=int)
        if 'c' in l:
            res[all_labels.index('c')] = 1
        elif 'v' in l:
            vl = l if self.diff_var_annotations else 'v'
            res[all_labels.index(vl)] = 1
        return res

    def get_matrix(self, f):
        adjm, nodes = self.mx_generator(f).getAdjMatrix(shuffle=True)
        nodes = np.asarray([self.int2onehot(lb, self.all_nodes) for lb in nodes])
        return self.normalize_adj_matrix_by_Kipf(adjm), nodes

    def get_batch(self, batch_size):
        f = True
        cd = list(self.cdata.values())
        while f:
            data = random.choice(cd)
            if len(data) >= batch_size:
                f = False

        samples = np.asarray(random.sample(data, batch_size))
        blabels = self.labels[samples].astype(int)
        nodes = []
        adjs = []
        for i in samples:
            a, n = self.get_matrix(self.data[i])
            adjs.append(a)
            nodes.append(n)

        nodes = np.asarray(nodes)
        adjs = np.asarray(adjs)

        return nodes, adjs, blabels

    def BatchGen(self, batchSize):
        for k in self.cdata:
            #print(f'size: {k}')
            subset = self.cdata[k].copy()
            random.shuffle(subset)
            for a in range(0, len(subset), batchSize):
                b = min(a + batchSize, len(subset))
                samples = np.arange(a, b)

                blabels = self.labels[samples].astype(int)
                nodes = []
                adjs = []
                for i in samples:
                    a, n = self.get_matrix(self.data[subset[i]])
                    adjs.append(a)
                    nodes.append(n)

                nodes = np.asarray(nodes)
                adjs = np.asarray(adjs)

                yield nodes, adjs, blabels


if __name__ == '__main__':
    ds = dataset(r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data\test.txt', '2d', 1)
    print('a')
    for _ in range(100):
        n, a, l = ds.get_batch(5)
        print('', end='')
    print('b')
