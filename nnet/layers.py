from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F

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
    def __init__(self, inDim, innerDim1, innerDim2, outDim, activation=F.relu, w1=None, b1=True, w2=None, b2=True,
                 w3=None, b3=True):
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
