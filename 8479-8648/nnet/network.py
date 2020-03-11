from torch.utils.tensorboard import SummaryWriter

from nnet.models import *

# from nnet.layers import *

'''
класс для простого использования сети
'''
T = 10


class network:
    def __init__(self, model, trw: SummaryWriter, tsw: SummaryWriter, lr=0.001, w_decay=5e-5):
        # print('cuda: {}'.format(torch.cuda.is_available()))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.dev)

        # self.model.emb = self.model.emb.to(self.dev)

        # if self.model.bodyNotRec is not None:
        #     self.model.bodyNotRec = self.model.bodyNotRec.to(self.dev)
        # if self.model.bodyRec is not None:
        #     self.model.bodyRec = self.model.bodyRec.to(self.dev)

        self.criteria = torch.nn.BCELoss().to(self.dev)
        # self.opt = torch.optim.SGD(self.model.parameters(True), lr=lr, weight_decay=w_decay)
        self.opt = torch.optim.Adam(self.model.parameters(True), lr=lr, weight_decay=w_decay)
        self.global_step = 0
        self.testWriter = tsw
        self.trainWriter = trw

    def train_step(self, adjs, labels, prnt=False):
        self.model.train(True)
        if not isinstance(labels, torch.Tensor):
            labels = torch.Tensor(labels).to(self.dev)
        if not isinstance(adjs, torch.Tensor):
            adjs = torch.Tensor(adjs).to(self.dev)
        self.opt.zero_grad()
        out = self.model(adjs)
        print(out.to('cpu').data[0], labels.mean().to('cpu').data)
        out = torch.sigmoid(out)
        loss = self.criteria(out, labels)
        loss.backward()
        self.opt.step()
        self.global_step += 1
        acc = np.mean(np.asarray(((out > 0.5) == labels).to('cpu')).astype(float))
        self.trainWriter.add_scalar('acc', float(acc), self.global_step)
        self.trainWriter.add_scalar('loss', float(loss), self.global_step)

        if prnt:
            print('step {} loss: {:.3f}, acc: {:.2f}'.format(self.global_step, loss.mean(), acc))

    def test(self, test_data, batch_size, ep):
        self.model.train(False)
        loss = 0.
        acc = 0.
        n = 0
        for adjs, labels in test_data.BatchGen(batch_size):
            if not isinstance(labels, torch.Tensor):
                labels = torch.Tensor(labels).to(self.dev)
            # if not isinstance(nodes, torch.Tensor):
            #     nodes = torch.Tensor(nodes).to(self.dev)
            if not isinstance(adjs, torch.Tensor):
                adjs = torch.Tensor(adjs).to(self.dev)

            out = torch.sigmoid(self.model(adjs, T))
            if torch.min(labels) < 0. or torch.max(labels) > 1. or torch.min(out) < 0. or torch.max(out) > 1.:
                print(torch.min(labels), torch.max(labels), torch.min(out), torch.max(out))
            loss += np.sum(np.asarray(self.criteria(out, labels).to('cpu').data)) * labels.size()[0]
            # print(torch.min(out), torch.max(out))
            acc += np.sum(np.asarray(((out > 0.5) == labels).to('cpu')).astype(float))
            n += labels.size()[0]

        self.testWriter.add_scalar('acc', float(acc / n), self.global_step)
        print('ep {} test accuracy: {:.3f}'.format(ep + 1, acc / n))
        self.testWriter.add_scalar('loss', float(loss / n), self.global_step)

    def save(self, pth):
        torch.save(self.model.state_dict(), pth)

    def load(self, pth):
        self.model.load_state_dict(torch.load(pth))


def make_net(trw, tsw, lr, wd, ldim):
    # if arch == 'test':
    #     rec = None
    #     nrec = ConvSeq(16, 16, 16, 12)
    #     ro = readout_GGNN(16, 16, 16, 16, True)
    #     net = ms.test_net(all_nodes, 16)

    net = test_nsat(ldim)

    return network(net, trw, tsw, lr, wd)
