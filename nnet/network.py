import numpy as np
from torch.utils.tensorboard import SummaryWriter

import nnet.models as ms
from nnet.layers import *
from nnet.models import ConvSeq

'''
класс для простого использования сети
'''


class network:
    def __init__(self, all_nodes, model, trw: SummaryWriter, tsw: SummaryWriter, lr=0.001):
        #print('cuda: {}'.format(torch.cuda.is_available()))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.dev)

        self.model.emb = self.model.emb.to(self.dev)

        if self.model.bodyNotRec is not None:
            self.model.bodyNotRec = self.model.bodyNotRec.to(self.dev)
        if self.model.bodyRec is not None:
            self.model.bodyRec = self.model.bodyRec.to(self.dev)

        self.criteria = torch.nn.BCELoss().to(self.dev)
        self.opt = torch.optim.SGD(self.model.parameters(True), lr=lr)
        self.global_step = 0
        self.testWriter = tsw
        self.trainWriter = trw

    def train_step(self, nodes, adjs, labels, prnt=False):
        self.model.train(True)
        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels).to(self.dev)
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.Tensor(nodes).to(self.dev)
        if not isinstance(adjs, torch.Tensor):
            adjs = torch.Tensor(adjs).to(self.dev)
        self.opt.zero_grad()
        out = torch.sigmoid(self.model(nodes, adjs))
        loss = self.criteria(out, labels)
        loss.backward()
        self.opt.step()
        self.global_step += 1
        acc = np.mean(np.asarray(((out > 0.5) == labels).to('cpu')).astype(float))
        self.trainWriter.add_scalar('acc', float(acc), self.global_step)
        self.trainWriter.add_scalar('loss', float(loss), self.global_step)

        if prnt:
            print('step {} loss: {:.3f}, acc: {:.2f}'.format(self.global_step, loss.mean(), acc))

    def test(self, test_data, batch_size):
        self.model.train(False)
        loss = 0.
        acc = 0.
        n = 0
        for nodes, adjs, labels in test_data.BatchGen(batch_size):
            if not isinstance(labels, torch.Tensor):
                labels = torch.Tensor(labels).to(self.dev)
            if not isinstance(nodes, torch.Tensor):
                nodes = torch.Tensor(nodes).to(self.dev)
            if not isinstance(adjs, torch.Tensor):
                adjs = torch.Tensor(adjs).to(self.dev)

            out = torch.sigmoid(self.model(nodes, adjs))
            loss += np.sum(np.asarray(self.criteria(out, labels).to('cpu').data))
            acc += np.sum(np.asarray(((out > 0.5) == labels).to('cpu')).astype(float))
            n += labels.size()[0]

        self.testWriter.add_scalar('acc', float(acc / n), self.global_step)
        print('test accuracy: {:.3f}'.format(acc/n))
        self.testWriter.add_scalar('loss', float(loss / n), self.global_step)

    def save(self, pth):
        torch.save(self.model.state_dict(), pth)

    def load(self, pth):
        self.model.load_state_dict(torch.load(pth))


def make_net(arch, ro, gt, all_nodes, trw, tsw,lr):
    if arch == 'test':
        rec = None
        nrec = ConvSeq(16, 16, 16, 12)
        ro = readout_GGNN(16, 16, 16, 16, True)
        net = ms.test_net(all_nodes, 16)

    return network(all_nodes, net, trw, tsw,lr)
