import os
from datetime import datetime
import numpy as np
import random

import torch
from torch.utils.tensorboard import SummaryWriter

from data import dataset
import nnet

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{},".format(0)

locdeb = True


def run_experiment(seed, print_rate, train_epochs, train_vars, test_vars, val_vars,
                   batch_size, test_batch_size, valid_batch_size,
                   data_path, logs_dir, save_dir,
                   train_only_positive, net_params, test_rnn_steps):
    # region params
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    optimizer = net_params['opt']
    loss = net_params['loss']
    classifier = net_params['classif']
    tnorm = net_params['tnorm']
    rec_bl = net_params['rnncell']
    hdim = net_params['hdim']
    iters = net_params['iters']
    lr = net_params['lr']
    w_decay = net_params['wdecay']
    gr_clip = net_params['grad_clip_ratio']
    arch = {
        'latent_dim': hdim,
        'std_T': iters,
        'classifier': classifier,
        'tnorm': tnorm,
        'tnorm_temperature': 0.5,
        'recurrent_block': rec_bl,
        'grad_clip': gr_clip
    }

    # endregion

    # region data
    if isinstance(train_vars, int):
        train_data_f = os.path.join(data_path, 'V{}Train.txt'.format(train_vars))
        print('loading train data')
        train_data = dataset(train_data_f, train_vars, train_only_positive)
    else:
        train_data = train_vars
    train_data.UpdBatchesSchedule(batch_size, seed)
    if isinstance(test_vars, int):
        test_data_f = os.path.join(data_path, 'V{}Test.txt'.format(test_vars))
        print('loading test data')
        test_data = dataset(test_data_f, test_vars, False)
    else:
        test_data = test_vars

    # endregion

    # region logs

    dtime = str(datetime.now())[:19].replace(':', '-')
    s = '{},seed={},hdim={},wd={},lr={},type={},rec_bl={}'.format(dtime, seed, hdim, w_decay, lr, classifier, rec_bl)
    if net_params['classif'] == 'CicruitSAT-like':
        s += ',tnorm={},loss={},only_pos={}'.format(tnorm, loss, 1 if train_only_positive else 0)
    print(s)

    if not os.path.exists(os.path.join(logs_dir, s, 'train')):
        os.makedirs(os.path.join(logs_dir, s, 'train'))
    trw = SummaryWriter(os.path.join(logs_dir, s, 'train'))

    if not os.path.exists(os.path.join(logs_dir, s, 'test')):
        os.makedirs(os.path.join(logs_dir, s, 'test'))
    tsw = SummaryWriter(os.path.join(logs_dir, s, 'test'))
    # endregion

    net = nnet.network(arch, optimizer, loss, lr, w_decay)
    #    make_net(trw, tsw, lr, w_decay, latent_dim,loss)
    i = 0
    for stps in test_rnn_steps:
        net.test(test_data, test_batch_size, 0, stps, tsw)
    for ep in range(train_epochs):
        print('epoch {}'.format(ep + 1))
        trb = train_data.BatchGen(batch_size)
        for adjs, labels in trb:
            net.train_step(adjs, labels, trw, (i % print_rate) == 0)
            i += 1
        for stps in test_rnn_steps:
            net.test(test_data, test_batch_size, ep, stps, tsw)

    if isinstance(val_vars, int):
        print('loading validation data')
        val_data_f = os.path.join(data_path, 'V{}Test.txt'.format(val_vars))
        val_data = dataset(val_data_f, val_vars, False)
    else:
        val_data = val_vars
    val_data.UpdBatchesSchedule(test_batch_size, seed)

    for stps in test_rnn_steps:
        net.test(val_data, valid_batch_size, train_epochs, stps, tsw, valid=True)
    net.save(os.path.join(save_dir, 'final_model.pt'))

    trw.flush()
    trw.close()

    tsw.flush()
    tsw.close()
    print('done')


def main():
    print_rate = 100
    batch_size = 64
    test_batch_size = 128
    valid_batch_size = 4
    only_pos = False
    hdim = 128

    train_epochs = 20
    data_path = '/mnt/nvme208/alex/sat/data2'
    logs_dir = '/mnt/nvme208/alex/sat/logs2'
    save_dir = '/mnt/nvme208/alex/sat/saves2'

    seeds = [13, 19, 42, 79, 97]
    test_rnn_steps = [26, 52, 78, 104]
    train_iters = 26
    rnn_cells = ['std_lstm', 'gru', 'ln_lstm']
    classifiers = ['NeuroSAT', 'CicruitSAT-like']

    if locdeb:
        train_epochs = 2
        data_path = r'C:\Users\Alex\Dropbox\институт\диссертация\статья нейросеть выводимость\data2'
        logs_dir = r'C:\ml_logs'
        save_dir = r'C:\ml_saves'
        test_rnn_steps = [10]
        hdim = 16
        train_iters = 10

    trainV = 9
    testV = 9
    valV = 10

    train_data_f = os.path.join(data_path, 'V(3-9)Train.txt')
    test_data_f = os.path.join(data_path, 'V(3-9)Test.txt')
    val_data_f = os.path.join(data_path, 'V10Test.txt')

    print('loading train data')
    train_data_pn = dataset(train_data_f, trainV, False)
    print('loading train data')
    train_data_p = dataset(train_data_f, trainV, True)
    print('loading test data')
    test_data = dataset(test_data_f, testV, False)
    print('loading validation data')
    val_data = dataset(val_data_f, valV, False)

    net_params = {'opt': ['SGD', 'Adam'][1], 'grad_clip_ratio': 0.65,
                  'hdim': hdim, 'iters': train_iters, 'lr': 2e-5,
                  'wdecay': 1e-10, 'tnorm': 'min'}

    net_params['classif'] = 'NeuroSAT'
    net_params['loss'] = 'bce'
    for rnncell in rnn_cells:
        net_params['rnncell'] = rnncell
        for seed in seeds:
            run_experiment(seed, print_rate, train_epochs,
                           train_data_pn,
                           test_data, val_data,
                           batch_size, test_batch_size, valid_batch_size,
                           data_path, logs_dir, save_dir,
                           False, net_params, test_rnn_steps)

    net_params['classif'] = 'CicruitSAT-like'
    net_params['tnorm'] = 'lukasievich'
    net_params['loss'] = 'bce'
    net_params['rnncell'] = 'ln_lstm'
    only_pos = False
    seed = 19
    run_experiment(seed, print_rate, train_epochs,
                   train_data_p if only_pos else train_data_pn,
                   test_data, val_data,
                   batch_size, test_batch_size, valid_batch_size,
                   data_path, logs_dir, save_dir,
                   only_pos, net_params, test_rnn_steps)

    for tnorm in ['min', 'product']:  # 2
        net_params['tnorm'] = tnorm
        for only_pos in [False, True]:  # 2
            net_params['loss'] = 'bce'
            for rnncell in rnn_cells:  # 3
                net_params['rnncell'] = rnncell
                for seed in seeds:  # 5
                    run_experiment(seed, print_rate, train_epochs,
                                   train_data_p if only_pos else train_data_pn,
                                   test_data, val_data,
                                   batch_size, test_batch_size, valid_batch_size,
                                   data_path, logs_dir, save_dir,
                                   only_pos, net_params, test_rnn_steps)


if __name__ == '__main__':
    main()
