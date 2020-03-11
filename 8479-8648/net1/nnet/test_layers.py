# from generator import make_graph_2partial, generateRandomCNFt, SolveSAT
# import numpy as np
#
# test_set_size = 1000
#
# test_batch_sz = 7
# vars = 4
#
# samples = {i: list() for i in range(25)}
#
# for i in range(test_set_size):
#     f = generateRandomCNFt(vars)
#     if len(f) == 0:
#         continue
#     g = make_graph_2partial(f)
#     l = SolveSAT(f)
#     samples[len(g.vertices)].append((f, g, l))
#
# mx_sz = 15
#
# graphs = samples[mx_sz][:test_batch_sz]
#
# all_nodes = set()
# for sz in samples:
#     gs = samples[sz]
#     for f, g, l in gs:
#         for v in g.vertices:
#             # вложения всех конъюнкций идинаковы
#             if 'v' in v:
#                 all_nodes.add(v)
#             if 'c' in v:
#                 all_nodes.add('c')
#
# all_nodes = sorted(list(all_nodes))
#
#
# def renorm_mx(mx):
#     D2 = np.diag(np.sqrt(np.sum(np.abs(mx), axis=0))) ** -1
#     D2[np.isinf(D2)] = 0.
#     m = np.matmul(np.matmul(D2, mx), D2) + np.eye(D2.shape[0])
#     return m
#
#
# def int2onehot(l, all_labels):
#     res = np.zeros(shape=[len(all_labels)], dtype=int)
#     if 'c' in l:
#         res[all_labels.index('c')] = 1
#     else:
#         res[all_labels.index(l)] = 1
#     return res
#
#
# ids = []
# matrices = []
# labels = []
# for f, g, l in graphs:
#     mx, lbls = g.getAdjMatrix()
#     mx = renorm_mx(mx)
#
#     one_hot_ids = np.asarray([int2onehot(lb, all_nodes) for lb in lbls])
#     labels.append(l)
#     ids.append(one_hot_ids)
#     matrices.append(mx)
#
# matrices = np.asarray(matrices)
# labels = np.asarray(labels).astype(int)
# ids = np.asarray(ids)

print()

from net1.nnet import network

from net1.data import dataset

testSet=dataset()


n= network.network(all_nodes)

for i in range(10000):
    n.train(ids,matrices,labels,i%100==0)