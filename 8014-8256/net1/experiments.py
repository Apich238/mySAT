import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from net1.data import dataset
from net1.nnet.network import make_net

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{},".format(1)

locdeb = False

graph_types = [ '2p','tree']
var_annot_lim = [1, 9]
read_out = ['GGNN']  # ['GGNN', 'IN', 'DTNN', 'My']
architectures = ['test']  # ['ConvNet', 'RecNet', 'ResConvNet', 'resRecNet']

if not locdeb:
    data_path = '/home/alex/sat/data'  # r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data'
else:
    data_path = r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data'  # r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data'

#data_path = '/home/alex/sat/data'  # r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data'

nVars = 5

train_data_f = os.path.join(data_path, 'V{}Train.txt'.format(nVars))
test_data_f = os.path.join(data_path, 'V{}Test.txt'.format(nVars))

train_epochs = 1

if not locdeb:
    train_epochs = 15
else:
    train_epochs = 5

batch_size = 64

if not locdeb:
    logs_dir = '/home/alex/sat/logs'
else:
    logs_dir = r'C:\ml_logs'

dtime = str(datetime.now())[:19].replace(':', '-')
print_rate = 20

lr=0.01

for var_lim in var_annot_lim:
    for gt in graph_types:
        print('loading train data')
        train_data = dataset(train_data_f, gt, var_lim)
        print('loading test data')
        test_data = dataset(test_data_f, gt, var_lim)
        # test_more_data = dataset(test_more_data_f, gt, var_lim)
        all_nodes = train_data.all_nodes.copy()
        for ro in read_out:
            for arch in architectures:
                # s = f'architecture {arch}, read out {ro}, variables similar {var_lim == 1}, graph type {gt}'
                s = '{}-ar_{},rOut_{},simVs_{},gt_{}'.format(dtime, arch, ro, var_lim == 1, gt)
                print(s)
                if not os.path.exists(os.path.join(logs_dir, s, 'train')):
                    os.makedirs(os.path.join(logs_dir, s, 'train'))
                trw = SummaryWriter(os.path.join(logs_dir, s, 'train'))

                if not os.path.exists(os.path.join(logs_dir, s, 'test')):
                    os.makedirs(os.path.join(logs_dir, s, 'test'))
                tsw = SummaryWriter(os.path.join(logs_dir, s, 'test'))

                net = make_net(arch, ro, gt, all_nodes, trw, tsw,lr)
                i = 0
                net.test(test_data, batch_size)
                for ep in range(train_epochs):
                    print('epoch {}'.format(ep))
                    for nodes, adjs, labels in train_data.BatchGen(batch_size):
                        net.train_step(nodes, adjs, labels, (i % print_rate) == 0)
                        i += 1
                    net.test(test_data, batch_size)
