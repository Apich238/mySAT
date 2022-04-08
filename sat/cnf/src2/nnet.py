import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

from torch.utils.tensorboard import SummaryWriter
# from torch.nn.parameter import Parameter
# from nnet.layers import *

import numpy as np
# from torch.nn.parameter import Parameter
# from nnet.layers import *
import torch
from torch.nn import Module, Parameter, LSTMCell, GRUCell, Linear
from ln_lstm import ln_LSTMCell2 as ln_LSTMCell

# from nnet import *

# from nnet.layers import *
'''
если в качестве биаса задать None, он не добавляется. Если True, создаётся новый. Если передать тензор, он используется как параметры
аналогично с весами, но без опции True
'''

'''
класс слоя для свёртки на графах, основан на определении Кипфа https://github.com/tkipf/pygcn
аналог свёртки 3*3
отличается тем, что работает с батчем графов, со множеством мариц смежности
батч делается по первому измерению
'''


# region layers
class GraphNeighbourConvolution(Module):
    def __init__(self, inDim, outDim, use_bias=True, weights=None, activation=None):
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
            torch.nn.init.uniform_(self.bias, -1., 1.)
        else:
            self.bias = use_bias
        self.activation = activation

        print()

    def forward(self, X, AdjsMxNorm):
        '''

        :param X: nodes features, format [b,n,f] where b is for batch,n is for nodes count,f is for features count, f=in_features
        :param AdjsMxNorm: normalized adjacency matrces, format [b,n,n], spersed tensor
        :return: new nodes features, format [b,n,f] where f in output dim
        '''

        weighted_features = torch.matmul(AdjsMxNorm, X)
        output = torch.matmul(weighted_features, self.weights)
        if self.bias is not None:
            output += self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


'''
просто слой персептрона на графе. фичи нод преобразуются без влияния соседей
аналог свёртки 1*1
'''


class GraphNodewiseConvolution(Module):
    def __init__(self, inDim, outDim, use_bias=True, weights=None, activation=None):
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
            torch.nn.init.uniform_(self.bias, -1., 1.)
        else:
            self.bias = None
        self.activation = activation

        print()

    def forward(self, X):
        '''

        :param X: nodes features, format [b,n,f] where b is for batch,n is for nodes count,f is for features count, f=in_features
        :return: new nodes features, format [b,n,f] where f in output dim
        '''

        output = torch.matmul(X, self.weights)
        if self.bias is not None:
            output += self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


'''
обычный блок ResNet со свёрткой для графа
'''


class GraphResidualBlock(Module):
    def __init__(self, inDim, innerDim, outDim, activation=F.relu, w1=None, b1=True, w2=None, b2=True):
        super().__init__()
        self.inDim = inDim
        self.innerDim = innerDim
        self.outDim = outDim
        self.activation = activation
        self.conv1 = GraphNeighbourConvolution(inDim, innerDim, b1, w1, activation)
        self.conv2 = GraphNeighbourConvolution(innerDim, outDim, b2, w2, None)

    def forward(self, X, AdjsMxNorm):
        output = self.conv1(X, AdjsMxNorm)
        output = self.conv2(output, AdjsMxNorm)
        output = self.activation(output + X)
        return output


class GraphBottleneckResidualBlock(Module):
    def __init__(self, inDim, innerDim1, innerDim2, outDim, activation=F.relu,
                 w1=None, b1=True, w2=None, b2=True, w3=None, b3=True):
        super().__init__()
        self.inDim = inDim
        self.innerDim1 = innerDim1
        self.innerDim2 = innerDim2
        self.outDim = outDim
        self.activation = activation
        self.conv1 = GraphNodewiseConvolution(inDim, innerDim1, b1, w1, self.activation)
        self.conv2 = GraphNeighbourConvolution(innerDim1, innerDim2, b2, w2, self.activation)
        self.conv3 = GraphNodewiseConvolution(innerDim2, outDim, b3, w3, None)

    def forward(self, X, AdjsMxNorm):
        output = self.conv1(X)
        output = self.conv2(output, AdjsMxNorm)
        output = self.conv3(output)
        output = self.activation(output + X)
        return output


class embedding(Module):
    def __init__(self, classes, feature_dim=16):
        super().__init__()
        clss = sorted(list(set(classes)))
        self.id_dict = {v: i for i, v in enumerate(clss)}
        self.embs = Parameter(torch.FloatTensor(len(clss), feature_dim))
        torch.nn.init.xavier_uniform(self.embs)

    def forward(self, ids):
        res = torch.matmul(ids, F.normalize(self.embs, p=2, dim=1))

        return res


'''
свёртки с двудольной матрицей
'''


# region readouts
# инфа про readout
# https://arxiv.org/pdf/1704.01212.pdf (4)
# https://arxiv.org/pdf/1511.05493.pdf (7)


class readout_GGNN(Module):
    def __init__(self, inDimt, inDim0, outDim, latentDim, j_tanh=True):
        super().__init__()
        self.i_net = self.i_netm(inDimt, inDim0, latentDim, outDim)
        self.j_net = self.j_netm(inDimt, latentDim, outDim)
        self.j_tanh = j_tanh

    def forward(self, ht, h0):
        iv = F.sigmoid(self.i_net.forward(ht, h0))
        jv = self.j_net.forward(ht)
        if self.j_tanh:
            jv = torch.tanh(jv)
        pr = iv * jv
        s = torch.sum(pr, dim=1)
        return s

    class i_netm(Module):
        def __init__(self, inDimt, inDim0, latentDim, outDim):
            super().__init__()

            self.w1 = Parameter(torch.FloatTensor(inDimt, latentDim))
            torch.nn.init.xavier_normal_(self.w1)
            self.b1 = Parameter(torch.FloatTensor(latentDim))
            torch.nn.init.normal_(self.b1)

            self.w2 = Parameter(torch.FloatTensor(inDim0, latentDim))
            torch.nn.init.xavier_normal_(self.w2)
            self.b2 = Parameter(torch.FloatTensor(latentDim))
            torch.nn.init.normal_(self.b2)

            self.w3 = Parameter(torch.FloatTensor(latentDim, outDim))
            torch.nn.init.xavier_normal_(self.w3)
            self.b3 = Parameter(torch.FloatTensor(outDim))
            torch.nn.init.normal_(self.b3)

        def forward(self, ht, h0):
            a = F.relu(torch.matmul(ht, self.w1) + self.b1)
            b = F.relu(torch.matmul(h0, self.w2) + self.b2)
            c = torch.matmul(a + b, self.w3) + self.b3
            return c

    class j_netm(Module):
        def __init__(self, inDimt, latentDim, outDim):
            super().__init__()
            self.w1 = Parameter(torch.FloatTensor(inDimt, latentDim))
            torch.nn.init.xavier_normal_(self.w1)
            self.b1 = Parameter(torch.FloatTensor(latentDim))
            torch.nn.init.normal_(self.b1)

            self.w3 = Parameter(torch.FloatTensor(latentDim, outDim))
            torch.nn.init.xavier_normal_(self.w3)
            self.b3 = Parameter(torch.FloatTensor(outDim))
            torch.nn.init.normal_(self.b3)

        def forward(self, ht):
            a = F.relu(torch.matmul(ht, self.w1) + self.b1)
            c = torch.matmul(a, self.w3) + self.b3
            return c

    # Battaglia, Peter, Pascanu, Razvan, Lai, Matthew, Rezende,
    # Danilo Jimenez, and Kavukcuoglu, Koray. Interaction networks
    # for learning about objects, relations and
    # physics. In Advances in Neural Information Processing
    # Systems, pp. 4502–4510, 2016.
    # https://arxiv.org/pdf/1612.00222.pdf


class readout_IN(Module):
    def __init__(self, inDimt, latentDim, outDim):
        super().__init__()
        self.w1 = Parameter(torch.FloatTensor(inDimt, latentDim))
        torch.nn.init.xavier_normal_(self.w1)
        self.b1 = Parameter(torch.FloatTensor(latentDim))
        torch.nn.init.normal_(self.b1)
        self.w2 = Parameter(torch.FloatTensor(latentDim, outDim))
        torch.nn.init.xavier_normal_(self.w2)
        self.b2 = Parameter(torch.FloatTensor(outDim))
        torch.nn.init.normal_(self.b2)

    def forward(self, ht, h0):
        s = torch.sum(ht, dim=1)
        x = F.relu(torch.matmul(s, self.w1) + self.b1)
        x = torch.matmul(x, self.w2) + self.b2
        return x

    # Convolutional Networks for Learning Molecular Fingerprints, Duvenaud et al. (2015)


class readout3(Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    # Deep Tensor Neural Networks, Schutt et al. ¨ (2017)
    # https://arxiv.org/pdf/1609.08259.pdf


class readout_DTNN(Module):
    def __init__(self, inDimt, latentDim, outDim):
        super().__init__()
        self.w1 = Parameter(torch.FloatTensor(inDimt, latentDim))
        torch.nn.init.xavier_normal_(self.w1)
        self.b1 = Parameter(torch.FloatTensor(latentDim))
        torch.nn.init.normal_(self.b1)
        self.w2 = Parameter(torch.FloatTensor(latentDim, outDim))
        torch.nn.init.xavier_normal_(self.w2)
        self.b2 = Parameter(torch.FloatTensor(outDim))
        torch.nn.init.normal_(self.b2)

    def forward(self, ht, h0):
        x = F.relu(torch.matmul(ht, self.w1) + self.b1)
        x = torch.matmul(x, self.w2) + self.b2
        s = torch.sum(x, dim=1)
        return s

    # Schutt, Kristof T, Arbabzadah, Farhad, Chmiela, Stefan, ¨
    # Muller, Klaus R, and Tkatchenko, Alexandre. Quantum- ¨
    # chemical insights from deep tensor neural networks. Nature Communications, 8, 2017.


class readout3(Module):
    def __init__(self):
        pass

    def forward(self):
        pass

    # The set2set model is specifically designed to operate on sets
    # and should have more expressive power than simply summing the
    # final node states. This model first applies a linear
    # projection to each tuple (hTv, xv) and then takes as input
    # the set of projected tuples T = {(hTv, xv)}. Then, after M
    # steps of computation, the set2set model produces a graph
    # level embedding q∗t which is invariant to the order of the of
    # the tuples T. We feed this embedding q∗t through a neural
    # network to produce the output.


class readout4(Module):
    def __init__(self):
        pass

    def forward(self):
        pass


# endregion


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
            torch.nn.init.xavier_uniform_(self.weights)
        else:
            self.weights = weights
        if not use_bias:
            self.bias = None
        else:
            self.bias = Parameter(torch.FloatTensor(outDim))
            torch.nn.init.zeros_(self.bias)

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
            l = batchMLP_layer(ind, oud, activation=torch.relu if i < Dpth - 1 else None, bn=use_bn)
            layers.append(l)
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, X):
        '''
        :param X: nodes features, format [b,n,f] where b is for batch,n is for nodes count,f is for features count, f=in_features
        :return: new nodes features, format [b,n,f] where f in output dim
        '''
        return self.mlp(X)


# class batchLSTM(Module):
#     def __init__(self, input_size, hidden_size, use_bn=False):
#         pass
#
#     def forward(self, x_t,h_t_1):
#         '''
#         :param X: nodes features, format [b,n,f] where b is for batch,n is for nodes count,f is for features count, f=in_features
#         :return: new nodes features, format [b,n,f] where f in output dim
#         '''
#         sg=torch.sigmoid
#         sc=torch.tanh
#
#         f_t=sg(self.W_f*x_t+self.U_f*h_t_1+self.b_f)
#         i_t=sg(self.W_i*x_t+self.U_i*h_t_1+self.b_i)
#         o_t=sg(self.W_o*x_t+self.U_o*h_t_1+self.b_o)
#
#         c_t=f_t*c_t1+i_t*sc(self.W_c*x_t+self.U_c*h_t_1+self.b_c)
#         h_t=o_t*sc(c_t)
#
#
#         return _,(_,_)


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
        cu = Cs + torch.tanh(self.Cu(torch.cat([Cs, MTLmsg], 2)))

        FL = Ls[:, np.arange(Ls.size()[1]).reshape((-1, 2))[:, ::-1].reshape(-1)]

        MCmsg = torch.matmul(Ms, self.Cmsg(cu))
        lu = Ls + torch.tanh(self.Lu(torch.cat([Ls, FL, MCmsg], 2)))
        return lu, cu


class test_nsat(Module):
    def __init__(self, dim):
        super().__init__()
        self.Cinit = torch.nn.Parameter(torch.FloatTensor(dim))
        torch.nn.init.normal_(self.Cinit)
        self.Linit = torch.nn.Parameter(torch.FloatTensor(dim))
        torch.nn.init.normal_(self.Linit)

        self.block = BiPartialTestBlock(dim, dim, 2 * dim, 2)
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

# endregion


opts = ['SGD', 'Adam']
losses = ['bce', 'mse', 'my_loss']

'''
класс для простого использования сети
'''


class network:
    def __init__(self, arch, opt, loss, lr=0.001, w_decay=5e-5):
        print('cuda: {}'.format('on' if torch.cuda.is_available() else 'off'))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mysat_net2(arch).to(self.dev)

        self._grad_clip = arch['grad_clip']

        if loss == 'bce':
            self.criteria = torch.nn.BCELoss(reduction='none').to(self.dev)
        elif loss == 'mse':
            self.criteria = torch.nn.MSELoss(reduction='none').to(self.dev)

        if opt == 'Adam':
            self.opt = torch.optim.Adam(self.model.parameters(True), lr=lr, weight_decay=w_decay)
        elif opt == 'SGD':
            self.opt = torch.optim.SGD(self.model.parameters(True), lr=lr, weight_decay=w_decay)

        self.global_step = 0
        # self.testWriter = tsw
        # self.trainWriter = trw

    def train_step(self, adjs, labels, trainWriter, prnt=False):
        self.model.train(True)
        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels).to(self.dev)
        if not isinstance(adjs, torch.Tensor):
            adjs = torch.Tensor(adjs).to(self.dev)
        self.opt.zero_grad()
        out, sat_solution = self.model(adjs)
        # print(out.to('cpu').data[0], labels.mean().to('cpu').data)
        # out = torch.sigmoid(out)
        loss = self.criteria(out, labels)

        # for i in np.where(np.logical_and(np.logical_not(labels.cpu().tolist()), (out.cpu() > 0.5).tolist()))[0]:
        #     with open('fars\\fars.txt', 'a') as ff:
        #         ff.write('\n\n')
        #         ff.write(str(adjs[i].data.numpy().tolist()) + '\n')
        #         ff.write(str(sat_solution[i].data.numpy().tolist()) + '\n')
        #         print(str(adjs[i].data.numpy().tolist()) + ' : ' + str(sat_solution[i].data.numpy().tolist()))

        loss = loss.mean()
        loss.backward()

        if self._grad_clip:
            torch.nn.utils.clip_grad_value_(self.model.parameters(True), self._grad_clip)

        self.opt.step()

        all_labels = np.asarray(labels.cpu().tolist()) > 0.5
        all_predictions = np.asarray(out.cpu().tolist()) > 0.5

        fars = np.logical_and(np.logical_not(all_labels), all_predictions)
        frrs = np.logical_and(all_labels, np.logical_not(all_predictions))
        accs = all_labels == all_predictions

        far = fars.mean()
        frr = frrs.mean()
        acc = accs.mean()

        trainWriter.add_scalar('train/acc', float(acc), self.global_step)
        trainWriter.add_scalar('train/far', float(far), self.global_step)
        trainWriter.add_scalar('train/frr', float(frr), self.global_step)
        trainWriter.add_scalar('train/loss', float(loss), self.global_step)

        if prnt:
            print(
                'step {: >6d}, acc={:.4f}, far={:.4f}, frr={:.4f}, loss={:.3f}'.format(self.global_step, acc, far, frr,
                                                                                       loss.mean()))
        self.global_step += 1

    def test(self, test_data, batch_size, ep, rnn_steps, testWriter, valid=0):
        self.model.train(False)
        n = 0

        all_labels = []
        all_losses = []
        all_predictions = []

        for adjs, labels in test_data.BatchGen(batch_size):
            if not isinstance(labels, torch.Tensor):
                labels = torch.Tensor(labels).to(self.dev)
            # if not isinstance(nodes, torch.Tensor):
            #     nodes = torch.Tensor(nodes).to(self.dev)
            if not isinstance(adjs, torch.Tensor):
                adjs = torch.Tensor(adjs).to(self.dev)

            out, satsolution = self.model(adjs, rnn_steps)
            if torch.min(labels) < 0. or torch.max(labels) > 1. or torch.min(out) < 0. or torch.max(out) > 1.:
                print(torch.min(labels), torch.max(labels), torch.min(out), torch.max(out))
            all_losses.extend(self.criteria(out, labels).cpu().tolist())
            # print(torch.min(out), torch.max(out))
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(out.cpu().tolist())

        all_losses = np.asarray(all_losses)
        all_labels = np.asarray(all_labels) > 0.5
        all_predictions = np.asarray(all_predictions) > 0.5

        fars = np.logical_and(np.logical_not(all_labels), all_predictions)
        frrs = np.logical_and(all_labels, np.logical_not(all_predictions))
        accs = all_labels == all_predictions

        far = fars.mean()
        frr = frrs.mean()
        acc = accs.mean()
        loss = all_losses.mean()

        if valid == 1:
            testWriter.add_scalar('validation_{}/acc'.format(test_data.nVars), acc, rnn_steps)
            testWriter.add_scalar('validation_{}/far'.format(test_data.nVars), far, rnn_steps)
            testWriter.add_scalar('validation_{}/frr'.format(test_data.nVars), frr, rnn_steps)
            testWriter.add_scalar('validation_{}/loss'.format(test_data.nVars), loss, rnn_steps)
            print('validation v={: >3d}, step={: >3d}: acc {:.3f}, far {:.3f}, frr {:.3f}, loss {:.3f}'.format(
                test_data.nVars, rnn_steps, acc, far, frr, loss))

            testWriter.add_scalar('reporting/validation_far_{}'.format(rnn_steps), far, self.global_step)
            testWriter.add_scalar('reporting/validation_frr_{}'.format(rnn_steps), frr, self.global_step)
            testWriter.add_scalar('reporting/validation_acc_{}'.format(rnn_steps), acc, self.global_step)

        elif valid == 2:
            testWriter.add_scalar('validation_{}_{}/acc'.format(test_data.nVars, self.global_step), acc, rnn_steps)
            testWriter.add_scalar('validation_{}_{}/far'.format(test_data.nVars, self.global_step), far, rnn_steps)
            testWriter.add_scalar('validation_{}_{}/frr'.format(test_data.nVars, self.global_step), frr, rnn_steps)
            testWriter.add_scalar('validation_{}_{}/loss'.format(test_data.nVars, self.global_step), loss, rnn_steps)
            print('validation v={: >3d}, step={: >3d}: acc {:.3f}, far {:.3f}, frr {:.3f}, loss {:.3f}'.format(
                test_data.nVars, rnn_steps, acc, far, frr, loss))

            testWriter.add_scalar('reporting/validation_far_{}'.format(rnn_steps), far, self.global_step)
            testWriter.add_scalar('reporting/validation_frr_{}'.format(rnn_steps), frr, self.global_step)
            testWriter.add_scalar('reporting/validation_acc_{}'.format(rnn_steps), acc, self.global_step)
        else:
            testWriter.add_scalar('test_{}/acc_s={}'.format(test_data.nVars, rnn_steps), acc, self.global_step)
            testWriter.add_scalar('test_{}/far_s={}'.format(test_data.nVars, rnn_steps), far, self.global_step)
            testWriter.add_scalar('test_{}/frr_s={}'.format(test_data.nVars, rnn_steps), frr, self.global_step)
            testWriter.add_scalar('test_{}/loss_s={}'.format(test_data.nVars, rnn_steps), loss, self.global_step)
            print(
                'ep {: >5d} test v={: >3d}, step={: >3d}: acc {:.3f}, far {:.3f}, frr {:.3f}, loss {:.3f}'.format(
                    ep + 1, test_data.nVars, rnn_steps, acc, far, frr, loss))

            testWriter.add_scalar('reporting/test_far_{}'.format(rnn_steps), far, self.global_step)
            testWriter.add_scalar('reporting/test_frr_{}'.format(rnn_steps), frr, self.global_step)
            testWriter.add_scalar('reporting/test_acc_{}'.format(rnn_steps), acc, self.global_step)

        testWriter.flush()

    def save(self, pth):
        torch.save(self.model.state_dict(), pth)

    def load(self, pth):
        self.model.load_state_dict(torch.load(pth))


class mysat_net(Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.ldim = arch['latent_dim']
        self.defaultSteps = arch['std_T']
        self.Cinit = torch.nn.Parameter(torch.FloatTensor(self.ldim))
        torch.nn.init.normal_(self.Cinit)
        self.Linit = torch.nn.Parameter(torch.FloatTensor(self.ldim))
        torch.nn.init.normal_(self.Linit)

        self.rec_block = arch['recurrent_block']

        if self.rec_block == 'test':
            # self.block = BiPartialTestBlock(self.ldim, self.ldim, 2 * self.ldim, 2)
            Ldim, Cdim, Hdim, Dpth = self.ldim, self.ldim, 2 * self.ldim, 2
            self.Cmsg = batchMLP(Cdim, Hdim, Cdim, Dpth, False)
            self.Lmsg = batchMLP(Ldim, Hdim, Ldim, Dpth, False)

            self.Cu = batchMLP(Cdim * 2, Hdim, Cdim, Dpth, False)
            self.Lu = batchMLP(Ldim * 3, Hdim, Ldim, Dpth, False)
        elif self.rec_block in ['std_lstm', 'ln_lstm', 'gru']:
            Ldim, Cdim, Hdim, Dpth = self.ldim, self.ldim, self.ldim, 4
            self.Cmsg = batchMLP(Cdim, Hdim, Cdim, Dpth, False)
            self.Lmsg = batchMLP(Ldim, Hdim, Ldim, Dpth, False)

            if self.rec_block == 'std_lstm':
                self.Cu = LSTMCell(self.ldim, self.ldim, True)
                self.Lu = LSTMCell(self.ldim * 2, self.ldim, True)
            elif self.rec_block == 'ln_lstm':
                self.Cu = ln_LSTMCell(self.ldim, self.ldim, True)
                self.Lu = ln_LSTMCell(self.ldim * 2, self.ldim, True)
            elif self.rec_block == 'gru':
                self.Cu = GRUCell(self.ldim, self.ldim, True)
                self.Lu = GRUCell(self.ldim * 2, self.ldim, True)

        self.cl = arch['classifier']
        if self.cl == 'NeuroSAT':
            self.Lvote = batchMLP(self.ldim, 2 * self.ldim, 1, 2, False)
        elif self.cl == 'CircuitSAT-like':
            self.tnormf = arch['tnorm']
            if 'tnorm_train' in arch:
                self.train_tnorm = arch['tnorm_train']
            else:
                self.train_tnorm = self.tnormf
            self.tnorm_tmp = arch['tnorm_temperature']
            self.train_temp = arch['temp_train']
            self.test_temp = arch['temp_test']
            self.Lvote = batchMLP(self.ldim, 2 * self.ldim, 1, 2, False)

    def _get_votes(self, l, Ms):

        if self.cl == 'NeuroSAT':
            lvotes = self.Lvote(l)
            y = torch.sigmoid(lvotes.mean((1, 2)))
        elif self.cl == 'CircuitSAT-like':
            lvotes = self.Lvote(l)

            if self.training:
                lvotes *= self.train_temp
            else:
                lvotes *= self.test_temp
            # lvotes=lvotes.view(size=(-1,1,lvotes.shape[1]))
            # align negatives
            # lvotes=torch.sigmoid(lvotes)
            lvotes = torch.softmax(lvotes.view(size=(lvotes.shape[0], -1, 2)), 2).view(size=(lvotes.shape[0], -1, 1))
            # cnf
            pr = lvotes * Ms

            if self.training:
                tnf = self.train_tnorm
            else:
                tnf = self.tnormf

            if tnf == 'min':
                disj = pr.max(1)[0]
                y = disj.min(1)[0]
            elif tnf == 'product':
                disj = 1 - (1 - pr).prod(1)
                y = disj.prod(1)
            elif tnf == 'lukasievich':
                disj = pr.sum(1).clamp_max(1)
                y = (disj.sum(1) - disj.shape[1] + 1).clamp_min(0)
            elif tnf == 'hamacher':
                inv = 1 - pr
                disj = 1 - inv.prod(1) / (1e-8 + inv.sum(1) - inv.prod(1))
                y = disj.prod(1) / (1e-8 + disj.sum(1) - disj.prod(1))
            elif tnf == 'circuit-sat':
                t = self.tnorm_tmp
                # or, smooth max
                pws = torch.exp(pr / t)
                disj = (pr * pws).sum(1) / (1e-8 + pws.sum(1))
                # and, smooth min
                pws = torch.exp(-disj / t)
                y = (disj * pws).sum(1) / (1e-8 + pws.sum(1))
            # diss=torch.matmul(lvotes,Ms)

        return y, lvotes

    def forward(self, Ms, T=None):
        if T is None:
            T = self.defaultSteps

        batchSZ = Ms.shape[0]
        literas = Ms.shape[1]
        clauses = Ms.shape[2]

        l = torch.stack([self.Linit] * (Ms.size()[1] * (Ms.size()[0]))).view(
            (Ms.size()[0], Ms.size()[1], self.Linit.size()[0]))
        lh = torch.zeros_like(l).view((batchSZ * literas, -1))

        c = torch.stack([self.Cinit] * (Ms.size()[2] * (Ms.size()[0]))).view(
            (Ms.size()[0], Ms.size()[2], self.Cinit.size()[0]))
        ch = torch.zeros_like(c).view((batchSZ * clauses, -1))

        FL = l[:, np.arange(l.size()[1]).reshape((-1, 2))[:, ::-1].reshape(-1)]

        i = 0
        cont = True
        while cont:
            if self.rec_block == 'test':
                MsT = Ms.permute((0, 2, 1))
                MTLmsg = torch.matmul(MsT, self.Lmsg(l))
                cu = c + torch.tanh(self.Cu(torch.cat([c, MTLmsg], 2)))

                FL = l[:, np.arange(l.size()[1]).reshape((-1, 2))[:, ::-1].reshape(-1)]

                MCmsg = torch.matmul(Ms, self.Cmsg(cu))
                lu = l + torch.tanh(self.Lu(torch.cat([l, FL, MCmsg], 2)))

                l, c = lu, cu  # self.block(l, c, Ms)
            elif self.rec_block in ['std_lstm', 'ln_lstm', 'gru']:
                MsT = Ms.transpose(1, 2)
                MTLmsg = torch.matmul(MsT, self.Lmsg(l))
                MTLmsg = MTLmsg.view((batchSZ * clauses, -1))
                if self.rec_block == 'gru':
                    ch = self.Cu(MTLmsg, c.view((batchSZ * clauses, -1)))
                    c = ch.view((batchSZ, clauses, -1))
                else:
                    c, ch = self.Cu(MTLmsg, (c.view((batchSZ * clauses, -1)), ch))
                    c = c.view((batchSZ, clauses, -1))

                MCmsg = torch.matmul(Ms, self.Cmsg(c))
                if self.rec_block == 'gru':
                    lh = self.Lu(torch.cat([FL, MCmsg], 2).view((batchSZ * literas, -1)),
                                 l.view((batchSZ * literas, -1)))
                    l = lh.view(batchSZ, literas, -1)
                else:
                    l, lh = self.Lu(torch.cat([FL, MCmsg], 2).view((batchSZ * literas, -1)),
                                    (l.view((batchSZ * literas, -1)), lh))
                    l = l.view(batchSZ, literas, -1)

            i += 1
            if T > 0:
                cont = i < T
            else:
                cont = i < 104
                # if cont:
                #     y, lvotes = self._get_votes(l, Ms)

        y, lvotes = self._get_votes(l, Ms)

        return y, lvotes


class mysat_net2(Module):
    def __init__(self, arch: dict):
        super().__init__()
        self.ldim = arch['latent_dim']
        self.defaultSteps = arch['std_T']
        self.Cinit = torch.nn.Parameter(torch.FloatTensor(self.ldim))
        torch.nn.init.normal_(self.Cinit)
        self.Linit = torch.nn.Parameter(torch.FloatTensor(self.ldim))
        torch.nn.init.normal_(self.Linit)

        self.rec_block = arch['recurrent_block']

        if self.rec_block == 'test':
            # self.block = BiPartialTestBlock(self.ldim, self.ldim, 2 * self.ldim, 2)
            Ldim, Cdim, Hdim, Dpth = self.ldim, self.ldim, 2 * self.ldim, 2
            self.Cmsg = batchMLP(Cdim, Hdim, Cdim, Dpth, False)
            self.Lmsg = batchMLP(Ldim, Hdim, Ldim, Dpth, False)

            self.Cu = batchMLP(Cdim * 2, Hdim, Cdim, Dpth, False)
            self.Lu = batchMLP(Ldim * 3, Hdim, Ldim, Dpth, False)
        elif self.rec_block in ['std_lstm', 'ln_lstm', 'gru']:
            Ldim, Cdim, Hdim, Dpth = self.ldim, self.ldim, self.ldim, 4
            self.Cmsg = batchMLP(Cdim, Hdim, Cdim, Dpth, False)
            self.Lmsg = batchMLP(Ldim, Hdim, Ldim, Dpth, False)

            if self.rec_block == 'std_lstm':
                self.Cu = LSTMCell(self.ldim, self.ldim, True)
                self.Lu = LSTMCell(self.ldim * 2, self.ldim, True)
            elif self.rec_block == 'ln_lstm':
                self.Cu = ln_LSTMCell(self.ldim, self.ldim, True)
                self.Lu = ln_LSTMCell(self.ldim * 2, self.ldim, True)
            elif self.rec_block == 'gru':
                self.Cu = GRUCell(self.ldim, self.ldim, True)
                self.Lu = GRUCell(self.ldim * 2, self.ldim, True)

        self.cl = arch['classifier']
        if self.cl == 'NeuroSAT':
            self.Lvote = batchMLP(self.ldim, 2 * self.ldim, 1, 2, False)
        elif self.cl == 'CircuitSAT-like':
            self.tnormf = arch['tnorm']
            if 'tnorm_train' in arch:
                self.train_tnorm = arch['tnorm_train']
            else:
                self.train_tnorm = self.tnormf
            self.tnorm_tmp = arch['tnorm_temperature']
            self.train_temp = arch['temp_train']
            self.test_temp = arch['temp_test']
            self.Lvote = batchMLP(self.ldim, 2 * self.ldim, 1, 2, False)

    def _get_votes(self, l, Ms):

        if self.cl == 'NeuroSAT':
            lvotes = self.Lvote(l)
            y = torch.sigmoid(lvotes.mean((1, 2)))
        elif self.cl == 'CircuitSAT-like':
            lvotes = self.Lvote(l)

            if self.training:
                lvotes *= self.train_temp
            else:
                lvotes *= self.test_temp
            # lvotes=lvotes.view(size=(-1,1,lvotes.shape[1]))
            # align negatives
            # lvotes=torch.sigmoid(lvotes)
            lvotes = torch.softmax(lvotes.view(size=(lvotes.shape[0], -1, 2)), 2).view(size=(lvotes.shape[0], -1, 1))
            # cnf
            pr = lvotes * Ms

            if self.training:
                tnf = self.train_tnorm
            else:
                tnf = self.tnormf

            if tnf == 'min':
                disj = pr.max(1)[0]
                y = disj.min(1)[0]
            elif tnf == 'product':
                disj = 1 - (1 - pr).prod(1)
                y = disj.prod(1)
            elif tnf == 'lukasievich':
                disj = pr.sum(1).clamp_max(1)
                y = (disj.sum(1) - disj.shape[1] + 1).clamp_min(0)
            elif tnf == 'hamacher':
                inv = 1 - pr
                disj = 1 - inv.prod(1) / (1e-8 + inv.sum(1) - inv.prod(1))
                y = disj.prod(1) / (1e-8 + disj.sum(1) - disj.prod(1))
            elif tnf == 'circuit-sat':
                t = self.tnorm_tmp
                # or, smooth max
                pws = torch.exp(pr / t)
                disj = (pr * pws).sum(1) / (1e-8 + pws.sum(1))
                # and, smooth min
                pws = torch.exp(-disj / t)
                y = (disj * pws).sum(1) / (1e-8 + pws.sum(1))
            # diss=torch.matmul(lvotes,Ms)

        return y, lvotes

    def forward(self, Ms, T=None):
        if T is None:
            T = self.defaultSteps

        batchSZ = Ms.shape[0]
        literas = Ms.shape[1]
        clauses = Ms.shape[2]

        lh = torch.stack([self.Linit] * (Ms.size()[1] * (Ms.size()[0]))).view(
            (Ms.size()[0], Ms.size()[1], self.Linit.size()[0]))
        l = torch.zeros_like(lh).view((batchSZ * literas, -1))

        ch = torch.stack([self.Cinit] * (Ms.size()[2] * (Ms.size()[0]))).view(
            (Ms.size()[0], Ms.size()[2], self.Cinit.size()[0]))
        c = torch.zeros_like(ch).view((batchSZ * clauses, -1))

        i = 0
        cont = True
        while cont:
            if self.rec_block == 'test':
                MsT = Ms.permute((0, 2, 1))
                MTLmsg = torch.matmul(MsT, self.Lmsg(lh))
                cu = ch + torch.tanh(self.Cu(torch.cat([ch, MTLmsg], 2)))

                FL = lh[:, np.arange(lh.size()[1]).reshape((-1, 2))[:, ::-1].reshape(-1)]

                MCmsg = torch.matmul(Ms, self.Cmsg(cu))
                lu = lh + torch.tanh(self.Lu(torch.cat([lh, FL, MCmsg], 2)))

                lh, ch = lu, cu  # self.block(l, ch, Ms)
            elif self.rec_block in ['std_lstm', 'ln_lstm', 'gru']:
                MsT = Ms.transpose(1, 2)
                MTLmsg = torch.matmul(MsT, self.Lmsg(lh))
                MTLmsg = MTLmsg.view((batchSZ * clauses, -1))
                if self.rec_block == 'gru':
                    c = self.Cu(MTLmsg, ch.view((batchSZ * clauses, -1)))
                    ch = c.view((batchSZ, clauses, -1))
                else:
                    ch, c = self.Cu(MTLmsg, (ch.view((batchSZ * clauses, -1)), c))
                    ch = ch.view((batchSZ, clauses, -1))

                MCmsg = torch.matmul(Ms, self.Cmsg(ch))
                FL = lh[:, np.arange(lh.size()[1]).reshape((-1, 2))[:, ::-1].reshape(-1)]

                if self.rec_block == 'gru':
                    l = self.Lu(torch.cat([FL, MCmsg], 2).view((batchSZ * literas, -1)),
                                lh.view((batchSZ * literas, -1)))
                    lh = l.view(batchSZ, literas, -1)
                else:

                    lh, l = self.Lu(torch.cat([FL, MCmsg], 2).view((batchSZ * literas, -1)),
                                    (lh.view((batchSZ * literas, -1)), l))
                    lh = lh.view(batchSZ, literas, -1)

            i += 1
            if T > 0:
                cont = i < T
            else:
                cont = i < 104
                # if cont:
                #     y, lvotes = self._get_votes(l, Ms)

        y, lvotes = self._get_votes(lh, Ms)

        return y, lvotes
