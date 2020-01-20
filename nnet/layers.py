from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
import torch.functional as F
import torch.nn.functional as nnf

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
        else:
            self.weights = weights
        if not use_bias:
            self.bias = None
        elif isinstance(use_bias, bool):
            self.bias = Parameter(torch.FloatTensor(outDim))
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
        if self.bias:
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
        else:
            self.weights = weights
        if not use_bias:
            self.bias = None
        elif isinstance(use_bias, bool):
            self.bias = Parameter(torch.FloatTensor(outDim))
        else:
            self.bias = use_bias
        self.activation = activation

        print()

    def forward(self, X):
        '''

        :param X: nodes features, format [b,n,f] where b is for batch,n is for nodes count,f is for features count, f=in_features
        :return: new nodes features, format [b,n,f] where f in output dim
        '''

        output = torch.matmul(X, self.weights)
        if self.bias:
            output += self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


'''
обычный блок ResNet со свёрткой для графа
'''


class GraphResidualBlock(Module):
    def __init__(self, inDim, innerDim, outDim, activation=nnf.relu, w1=None, b1=True, w2=None, b2=True):
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
    def __init__(self, inDim, innerDim1, innerDim2, outDim, activation=nnf.relu, w1=None, b1=True, w2=None, b2=True,
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
