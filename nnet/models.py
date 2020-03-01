from torch.nn.modules import ModuleList

from nnet.layers import *


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


class a_classifier(Module):
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