# from torch.nn.parameter import Parameter
# from nnet.layers import *

import numpy as np
# from torch.nn.parameter import Parameter
# from nnet.layers import *
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter


# def get_net_module(name, all_nodes):
#     if name == 'test':
#         return TestNet(all_nodes)
#

# class TestNet(Module):
#     def __init__(self, all_nodes):
#         super().__init__()
#
#         self.emb = embedding(all_nodes, 16)
#         self.conv1 = GraphNeighbourConvolution(16, 16, True)
#         self.conv2 = GraphNeighbourConvolution(16, 16, True)
#         self.conv3 = GraphNeighbourConvolution(16, 16, True)
#         self.conv4 = GraphNeighbourConvolution(16, 16, True)
#         self.conv5 = GraphNeighbourConvolution(16, 16, True)
#         self.conv6 = GraphNeighbourConvolution(16, 16, True)
#         self.conv7 = GraphNeighbourConvolution(16, 16, True)
#         self.conv8 = GraphNeighbourConvolution(16, 16, True)
#         self.rout = readout_GGNN(16, 16, 16)
#         self.cl_weights = Parameter(torch.FloatTensor(16, 1))
#         torch.nn.init.xavier_normal_(self.cl_weights)
#
#     def forward(self, nodes, adjMs):
#
#         if not isinstance(nodes, torch.Tensor):
#             nodes = torch.Tensor(nodes)
#         if not isinstance(adjMs, torch.Tensor):
#             adjMs = torch.Tensor(adjMs)
#         x0 = self.emb.forward(nodes)
#         x = F.leaky_relu(self.conv1.forward(x0, adjMs))
#         x = F.leaky_relu(self.conv2.forward(x, adjMs))
#         x = F.leaky_relu(self.conv3.forward(x, adjMs))
#         x = F.leaky_relu(self.conv4.forward(x, adjMs))
#         x = F.leaky_relu(self.conv5.forward(x, adjMs))
#         x = F.leaky_relu(self.conv6.forward(x, adjMs))
#         x = F.leaky_relu(self.conv7.forward(x, adjMs))
#         x = F.leaky_relu(self.conv8.forward(x, adjMs))
#         x = self.rout.forward(x, x0)
#         x = torch.matmul(x, self.cl_weights)
#         return x

# class ConvSeq(Module):
#     def __init__(self, dim_in, dim_latent, dim_out, layers=1, activation=F.relu):
#         super().__init__()
#         self.layers = ModuleList()
#         for i in range(layers):
#             di = dim_in if i == 0 else dim_latent
#             do = dim_out if i == layers - 1 else dim_latent
#             l = GraphNeighbourConvolution(di, do, True, activation=activation)
#             self.layers.append(l)
#
#     def forward(self, ht, adjs):
#         for l in self.layers:
#             ht = l(ht, adjs)
#         return ht
#
#
# class test_net(Module):
#     def __init__(self, all_nodes, node_features=16):
#         super().__init__()
#         self.emb = embedding(all_nodes, node_features)
#         self.bodyNotRec = ConvSeq(16, 16, 16, 12)
#         self.bodyRec = None
#         self.ro = readout_GGNN(16, 16, 16, 16, True)
#         self.robn = torch.nn.BatchNorm1d(node_features)
#         self.cl = torch.nn.Linear(node_features, 1, bias=False)
#
#     def forward(self, nodes, adjs, rec_times=0):
#         h0 = nodes
#         h0 = self.emb(h0)
#         ht = h0
#         if self.bodyNotRec is not None:
#             ht = self.bodyNotRec(ht, adjs=adjs)
#         if self.bodyRec is not None:
#             for i in range(rec_times):
#                 ht = self.bodyRec(X=ht, AdjsMxNorm=adjs)
#         r = self.ro(ht, h0)
#         if r.shape[0] > 1:
#             r = self.robn(r)
#         r = self.cl(r)[:, 0]
#         return r


class batchMLP_layer(Module):
    def __init__(self, inDim, outDim, use_bias=True, weights=None, activation=None, bn=False):
        super().__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.weights = None
        if weights is None:
            self.weights = Parameter(torch.FloatTensor(inDim, outDim))
            torch.nn.init.xavier_uniform(self.weights)
        else:
            self.weights = weights
        if not use_bias:
            self.bias = None
        elif isinstance(use_bias, bool):
            self.bias = Parameter(torch.FloatTensor(outDim))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

        self.bn = None
        if bn:
            self.bn = torch.nn.BatchNorm1d(outDim)
        self.activation = activation

    def forward(self, X):
        '''

        :param X: nodes features, format [b,n,f] where b is for batch,n is for nodes count,f is for features count, f=in_features
        :return: new nodes features, format [b,n,f] where f in output dim
        '''

        output = torch.matmul(X, self.weights)
        if self.bias is not None:
            output += self.bias
        if self.bn is not None:
            tmp = output.view(-1, output.size()[2])
            tmp = self.bn(tmp)
            output = tmp.view(output.size())

            # output=self.bn(output.unsqueeze(3))
        if self.activation is not None:
            output = self.activation(output)
        return output


class batchMLP(Module):
    def __init__(self, iDim, hDim, oDim, Dpth, use_bn=False):
        super().__init__()
        self.inDim = iDim
        self.outDim = oDim
        layers = []
        for i in range(Dpth):
            ind = iDim if i == 0 else hDim
            oud = oDim if i == Dpth - 1 else hDim
            layers.append(batchMLP_layer(ind, oud, activation=F.relu if i < Dpth - 1 else None, bn=use_bn))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, X):
        '''
        :param X: nodes features, format [b,n,f] where b is for batch,n is for nodes count,f is for features count, f=in_features
        :return: new nodes features, format [b,n,f] where f in output dim
        '''
        return self.mlp(X)


class BiPartialTestBlock(Module):
    def __init__(self, Ldim, Cdim, Hdim, Dpth=2):
        super().__init__()
        self.Cmsg = batchMLP(Cdim, Hdim, Cdim, Dpth, False)
        self.Lmsg = batchMLP(Ldim, Hdim, Ldim, Dpth, False)

        self.Cu = batchMLP(Cdim * 2, Hdim, Cdim, Dpth, False)
        self.Lu = batchMLP(Ldim * 3, Hdim, Ldim, Dpth, False)


    def forward(self, Ls, Cs, Ms):
        MsT = Ms.permute((0, 2, 1))
        MTLmsg = torch.matmul(MsT, self.Lmsg(Ls))
        cu = Cs + F.tanh(self.Cu(torch.cat([Cs, MTLmsg], 2)))

        FL = Ls[:, np.arange(Ls.size()[1]).reshape((-1, 2))[:, ::-1].reshape(-1)]

        MCmsg = torch.matmul(Ms, self.Cmsg(cu))
        lu = Ls + F.tanh(self.Lu(torch.cat([Ls, FL, MCmsg], 2)))
        return lu, cu


class test_nsat(Module):
    def __init__(self, dim):
        super().__init__()
        self.Cinit = torch.nn.Parameter(torch.FloatTensor(dim))
        torch.nn.init.normal_(self.Cinit)
        self.Linit = torch.nn.Parameter(torch.FloatTensor(dim))
        torch.nn.init.normal_(self.Linit)

        self.block = BiPartialTestBlock(dim, dim, 2 * dim, 1)
        self.Lvote = batchMLP(dim, 2 * dim, 1, 2, False)  # True)

    def forward(self, Ms, T=10):
        l = torch.stack([self.Linit] * (Ms.size()[1] * (Ms.size()[0]))).view(
            (Ms.size()[0], Ms.size()[1], self.Linit.size()[0]))
        c = torch.stack([self.Cinit] * (Ms.size()[2] * (Ms.size()[0]))).view(
            (Ms.size()[0], Ms.size()[2], self.Cinit.size()[0]))
        for i in range(T):
            l, c = self.block(l, c, Ms)
        lvotes = self.Lvote(l)
        y = lvotes.mean((1, 2))
        return y

# class a1_classifier(Module):
#     def __init__(self, readout, all_nodes, node_features=16, bodyNotRec=None, bodyRec=None):
#         super().__init__()
#         self.emb = embedding(all_nodes, node_features)
#         self.bodyNotRec = bodyNotRec
#         self.bodyRec = bodyRec
#         self.ro = readout
#         self.robn = torch.nn.BatchNorm1d(node_features)
#         self.cl = torch.nn.Linear(node_features, 1, bias=False)
#
#     def forward(self, nodes, adjs, rec_times=0):
#         h0 = nodes
#         h0 = self.emb(h0)
#         ht = h0
#         if self.bodyNotRec is not None:
#             ht = self.bodyNotRec(ht, adjs=adjs)
#         if self.bodyRec is not None:
#             for i in range(rec_times):
#                 ht = self.bodyRec(X=ht, AdjsMxNorm=adjs)
#         r = self.ro(ht, h0)
#         if r.shape[0] > 1:
#             r = self.robn(r)
#         r = self.cl(r)[:, 0]
#         return r
#
#
# class a2_classifier(Module):
#     def __init__(self, readout, all_nodes, node_features=16, bodyNotRec=None, bodyRec=None):
#         super().__init__()
#         self.emb = embedding(all_nodes, node_features)
#         self.bodyNotRec = bodyNotRec
#         self.bodyRec = bodyRec
#         self.ro = readout
#         self.robn = torch.nn.BatchNorm1d(node_features)
#         self.cl = torch.nn.Linear(node_features, 1, bias=False)
#
#     def forward(self, nodes, adjs, rec_times=0):
#         h0 = nodes
#         h0 = self.emb(h0)
#         ht = h0
#         if self.bodyNotRec is not None:
#             ht = self.bodyNotRec(ht, adjs=adjs)
#         if self.bodyRec is not None:
#             for i in range(rec_times):
#                 ht = self.bodyRec(X=ht, AdjsMxNorm=adjs)
#         r = self.ro(ht, h0)
#         if r.shape[0] > 1:
#             r = self.robn(r)
#         r = self.cl(r)[:, 0]
#         return r
#
#
# class a4_classifier(Module):
#     def __init__(self, readout, all_nodes, node_features=16, bodyNotRec=None, bodyRec=None):
#         super().__init__()
#         self.emb = embedding(all_nodes, node_features)
#         self.bodyNotRec = bodyNotRec
#         self.bodyRec = bodyRec
#         self.ro = readout
#         self.robn = torch.nn.BatchNorm1d(node_features)
#         self.cl = torch.nn.Linear(node_features, 1, bias=False)
#
#     def forward(self, nodes, adjs, rec_times=0):
#         h0 = nodes
#         h0 = self.emb(h0)
#         ht = h0
#         if self.bodyNotRec is not None:
#             ht = self.bodyNotRec(ht, adjs=adjs)
#         if self.bodyRec is not None:
#             for i in range(rec_times):
#                 ht = self.bodyRec(X=ht, AdjsMxNorm=adjs)
#         r = self.ro(ht, h0)
#         if r.shape[0] > 1:
#             r = self.robn(r)
#         r = self.cl(r)[:, 0]
#         return r
