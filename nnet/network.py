import numpy as np
import torch

from nnet.layers import *
from nnet.models import a_classifier, ConvSeq

'''
класс для простого использования сети
'''


class network:
    def __init__(self, all_nodes, model, lr=0.0001):
        self.model = model
        self.criteria = torch.nn.BCELoss()
        self.opt = torch.optim.SGD(self.model.parameters(True), lr=lr)
        self.global_step = 0

    def train_step(self, nodes, adjs, labels, prnt=False):
        self.model.train(True)
        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels)
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.Tensor(nodes)
        if not isinstance(adjs, torch.Tensor):
            adjs = torch.Tensor(adjs)
        self.opt.zero_grad()
        out = torch.sigmoid(self.model(nodes, adjs))
        loss = self.criteria(out, labels)
        loss.backward()
        self.opt.step()
        self.global_step += 1
        acc = np.mean(np.asarray((out > 0.5) == labels).astype(float))

        if prnt:
            print(f'step {self.global_step} loss: {loss.mean():.3}, acc: {acc:.2}')

    def test(self, nodes, adjs, labels):
        self.model.train(False)
        pass

    def save(self, pth):
        torch.save(self.model.state_dict(), pth)

    def load(self, pth):
        self.model.load_state_dict(torch.load(pth))


def make_net(arch, ro, gt, all_nodes, logs_dir):
    if arch == 'test':
        rec = None
        nrec = ConvSeq(16, 16, 16, 12)
        ro = readout_GGNN(16, 16, 16, 16, True)

    net = a_classifier(ro, all_nodes, 16, nrec, rec)
    return network(all_nodes, net)
