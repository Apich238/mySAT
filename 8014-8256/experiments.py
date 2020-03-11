import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from data import dataset
from nnet.network import make_net

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{},".format(0)

locdeb = False

# graph_types = [ '2p','tree']
# var_annot_lim = [1, 9]
# read_out = ['GGNN']  # ['GGNN', 'IN', 'DTNN', 'My']
# architectures = ['test']  # ['ConvNet', 'RecNet', 'ResConvNet', 'resRecNet']

if not locdeb:
    data_path = '/home/alex/sat/data'
else:
    data_path = r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data'

nVars = 5

train_data_f = os.path.join(data_path, 'V{}Train.txt'.format(nVars))
test_data_f = os.path.join(data_path, 'V{}Test.txt'.format(nVars))

train_epochs = 1

if not locdeb:
    train_epochs = 8
else:
    train_epochs = 15

batch_size = 64

if not locdeb:
    logs_dir = '/home/alex/sat/logs'
else:
    logs_dir = r'C:\ml_logs'

dtime = str(datetime.now())[:19].replace(':', '-')
print_rate = 20
lr = 0.00001
w_decay = 0.00001
ldim = 32

s = '{},wd={},lr={}_c={}'.format(dtime, w_decay, lr,
                                 "no-rec,with-mistake,no-bn,1_hidden_everywhere,lvote_2,init_xavier_zero,tanh")
print(s)

print('loading train data')
train_data = dataset(train_data_f, nVars)
print('loading test data')
test_data = dataset(test_data_f, nVars)
# test_more_data = dataset(test_more_data_f, gt, var_lim)
# all_nodes = train_data.all_nodes.copy()
# for ro in read_out:

# s = f'architecture {arch}, read out {ro}, variables similar {var_lim == 1}, graph type {gt}'

# region data
if not os.path.exists(os.path.join(logs_dir, s, 'train')):
    os.makedirs(os.path.join(logs_dir, s, 'train'))
trw = SummaryWriter(os.path.join(logs_dir, s, 'train'))
if not os.path.exists(os.path.join(logs_dir, s, 'test')):
    os.makedirs(os.path.join(logs_dir, s, 'test'))
# endregion
tsw = SummaryWriter(os.path.join(logs_dir, s, 'test'))
net = make_net(trw, tsw, lr, w_decay, ldim)
i = 0
net.test(test_data, batch_size, 0)
for ep in range(train_epochs):
    print('epoch {}'.format(ep))
    for adjs, labels in train_data.BatchGen(batch_size):
        net.train_step(adjs, labels, (i % print_rate) == 0)
        i += 1
    net.test(test_data, batch_size, ep)
net.test(test_data, batch_size, train_epochs)

tsw.flush()
trw.flush()
print('done')
