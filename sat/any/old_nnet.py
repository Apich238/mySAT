
import torch
from torch.nn import Module,Parameter,functional as F

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



# endregion


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


def get_net_module(name, all_nodes):
    if name == 'test':
        return TestNet(all_nodes)


class TestNet(Module):
    def __init__(self, all_nodes):
        super().__init__()

        self.emb = embedding(all_nodes, 16)
        self.conv1 = GraphNeighbourConvolution(16, 16, True)
        self.conv2 = GraphNeighbourConvolution(16, 16, True)
        self.conv3 = GraphNeighbourConvolution(16, 16, True)
        self.conv4 = GraphNeighbourConvolution(16, 16, True)
        self.conv5 = GraphNeighbourConvolution(16, 16, True)
        self.conv6 = GraphNeighbourConvolution(16, 16, True)
        self.conv7 = GraphNeighbourConvolution(16, 16, True)
        self.conv8 = GraphNeighbourConvolution(16, 16, True)
        self.rout = readout_GGNN(16, 16, 16)
        self.cl_weights = Parameter(torch.FloatTensor(16, 1))
        torch.nn.init.xavier_normal_(self.cl_weights)

    def forward(self, nodes, adjMs):

        if not isinstance(nodes, torch.Tensor):
            nodes = torch.Tensor(nodes)
        if not isinstance(adjMs, torch.Tensor):
            adjMs = torch.Tensor(adjMs)
        x0 = self.emb.forward(nodes)
        x = F.leaky_relu(self.conv1.forward(x0, adjMs))
        x = F.leaky_relu(self.conv2.forward(x, adjMs))
        x = F.leaky_relu(self.conv3.forward(x, adjMs))
        x = F.leaky_relu(self.conv4.forward(x, adjMs))
        x = F.leaky_relu(self.conv5.forward(x, adjMs))
        x = F.leaky_relu(self.conv6.forward(x, adjMs))
        x = F.leaky_relu(self.conv7.forward(x, adjMs))
        x = F.leaky_relu(self.conv8.forward(x, adjMs))
        x = self.rout.forward(x, x0)
        x = torch.matmul(x, self.cl_weights)
        return x

class ConvSeq(Module):
    def __init__(self, dim_in, dim_latent, dim_out, layers=1, activation=F.relu):
        super().__init__()
        self.layers = ModuleList()
        for i in range(layers):
            di = dim_in if i == 0 else dim_latent
            do = dim_out if i == layers - 1 else dim_latent
            l = GraphNeighbourConvolution(di, do, True, activation=activation)
            self.layers.append(l)

    def forward(self, ht, adjs):
        for l in self.layers:
            ht = l(ht, adjs)
        return ht


class test_net(Module):
    def __init__(self, all_nodes, node_features=16):
        super().__init__()
        self.emb = embedding(all_nodes, node_features)
        self.bodyNotRec = ConvSeq(16, 16, 16, 12)
        self.bodyRec = None
        self.ro = readout_GGNN(16, 16, 16, 16, True)
        self.robn = torch.nn.BatchNorm1d(node_features)
        self.cl = torch.nn.Linear(node_features, 1, bias=False)

    def forward(self, nodes, adjs, rec_times=0):
        h0 = nodes
        h0 = self.emb(h0)
        ht = h0
        if self.bodyNotRec is not None:
            ht = self.bodyNotRec(ht, adjs=adjs)
        if self.bodyRec is not None:
            for i in range(rec_times):
                ht = self.bodyRec(X=ht, AdjsMxNorm=adjs)
        r = self.ro(ht, h0)
        if r.shape[0] > 1:
            r = self.robn(r)
        r = self.cl(r)[:, 0]
        return r





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


class a1_classifier(Module):
    def __init__(self, readout, all_nodes, node_features=16, bodyNotRec=None, bodyRec=None):
        super().__init__()
        self.emb = embedding(all_nodes, node_features)
        self.bodyNotRec = bodyNotRec
        self.bodyRec = bodyRec
        self.ro = readout
        self.robn = torch.nn.BatchNorm1d(node_features)
        self.cl = torch.nn.Linear(node_features, 1, bias=False)

    def forward(self, nodes, adjs, rec_times=0):
        h0 = nodes
        h0 = self.emb(h0)
        ht = h0
        if self.bodyNotRec is not None:
            ht = self.bodyNotRec(ht, adjs=adjs)
        if self.bodyRec is not None:
            for i in range(rec_times):
                ht = self.bodyRec(X=ht, AdjsMxNorm=adjs)
        r = self.ro(ht, h0)
        if r.shape[0] > 1:
            r = self.robn(r)
        r = self.cl(r)[:, 0]
        return r


class a2_classifier(Module):
    def __init__(self, readout, all_nodes, node_features=16, bodyNotRec=None, bodyRec=None):
        super().__init__()
        self.emb = embedding(all_nodes, node_features)
        self.bodyNotRec = bodyNotRec
        self.bodyRec = bodyRec
        self.ro = readout
        self.robn = torch.nn.BatchNorm1d(node_features)
        self.cl = torch.nn.Linear(node_features, 1, bias=False)

    def forward(self, nodes, adjs, rec_times=0):
        h0 = nodes
        h0 = self.emb(h0)
        ht = h0
        if self.bodyNotRec is not None:
            ht = self.bodyNotRec(ht, adjs=adjs)
        if self.bodyRec is not None:
            for i in range(rec_times):
                ht = self.bodyRec(X=ht, AdjsMxNorm=adjs)
        r = self.ro(ht, h0)
        if r.shape[0] > 1:
            r = self.robn(r)
        r = self.cl(r)[:, 0]
        return r


class a4_classifier(Module):
    def __init__(self, readout, all_nodes, node_features=16, bodyNotRec=None, bodyRec=None):
        super().__init__()
        self.emb = embedding(all_nodes, node_features)
        self.bodyNotRec = bodyNotRec
        self.bodyRec = bodyRec
        self.ro = readout
        self.robn = torch.nn.BatchNorm1d(node_features)
        self.cl = torch.nn.Linear(node_features, 1, bias=False)

    def forward(self, nodes, adjs, rec_times=0):
        h0 = nodes
        h0 = self.emb(h0)
        ht = h0
        if self.bodyNotRec is not None:
            ht = self.bodyNotRec(ht, adjs=adjs)
        if self.bodyRec is not None:
            for i in range(rec_times):
                ht = self.bodyRec(X=ht, AdjsMxNorm=adjs)
        r = self.ro(ht, h0)
        if r.shape[0] > 1:
            r = self.robn(r)
        r = self.cl(r)[:, 0]
        return r

# endregion
