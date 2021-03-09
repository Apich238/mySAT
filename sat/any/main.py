import os
from datetime import datetime
import itertools

from joblib import delayed,Parallel

from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset import TreeFormulasDataset
from nnet import SimpleTreeSAT


def run_experiment(train_dataset, test_dataset, validation_dataset, batch_sz, p, epochs, rnn_steps, dim, cl_type, opt,
                   lr, momentum, wdecay, opt_eps, nesterov, seed, use_cuda, logdir, grad_clip=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    global_step = 0

    os.makedirs(logdir)

    device = use_cuda

    if use_cuda and use_cuda.startswith('cuda') and not torch.cuda.is_available():
        device = False

    # use_cuda = use_cuda and True#torch.cuda.is_available()

    if not device:
        device = torch.device('cpu:0')
    else:
        device = torch.device(device)

    net = SimpleTreeSAT(dim, classifier_type=cl_type).to(device)
    lossf = torch.nn.BCELoss().to(device)

    if opt == 'sgd':
        opt = torch.optim.SGD(net.parameters(), lr, momentum, weight_decay=wdecay, nesterov=nesterov)
    elif opt == 'adam':
        opt = torch.optim.Adam(net.parameters(), lr, eps=opt_eps, weight_decay=wdecay)

    train_loader = DataLoader(train_dataset, batch_sz, True)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_sz, shuffle=False)

    logger = SummaryWriter(logdir)

    print('experiment:', p)

    def train():
        nonlocal global_step
        net.train()
        for i, batch in enumerate(train_loader):
            opt.zero_grad()
            res = net(mxs=batch['matrix'].to(device),
                      cons=batch['conops'].to(device),
                      negs=batch['negops'].to(device),
                      rnn_steps=rnn_steps)
            loss = lossf(res, batch['label'].to(dtype=torch.float32, device=device))

            r = res.cpu() > 0.5
            lbl = batch['label'].to(torch.bool)
            tp = torch.sum(r * lbl).cpu().data.numpy().tolist()
            tn = torch.sum((~r) * (~lbl)).cpu().data.numpy().tolist()
            fp = torch.sum(r * (~lbl)).cpu().data.numpy().tolist()
            fn = torch.sum((~r) * lbl).cpu().data.numpy().tolist()
            l = loss.cpu().data.numpy().tolist()

            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_value_(net.parameters(True), grad_clip)

            opt.step()

            logger.add_scalar('train/acc', (tp + tn) / len(lbl), global_step)
            logger.add_scalar('train/fp', (fp) / len(lbl), global_step)
            logger.add_scalar('train/fn', (fn) / len(lbl), global_step)
            logger.add_scalar('train/loss', (l), global_step)

            global_step += 1
            if i % 20 == 0 and i > 0:
                print('step', i, ':', 'acc', (tp + tn) / len(lbl), 'fp', fp / len(lbl), 'fn', fn / len(lbl), 'loss', l)

    def test(dsl: DataLoader, name):
        tp, tn, fp, fn = 0, 0, 0, 0
        net.eval()
        for batch in dsl:
            res = net(mxs=batch['matrix'].to(device),
                      cons=batch['conops'].to(device),
                      negs=batch['negops'].to(device), rnn_steps=rnn_steps)
            res = res.cpu() > 0.5
            lbl = batch['label'].to(torch.bool)

            tp += torch.sum(res * lbl).data.numpy().tolist()
            tn += torch.sum((~res) * (~lbl)).data.numpy().tolist()
            fp += torch.sum(res * (~lbl)).data.numpy().tolist()
            fn += torch.sum((~res) * lbl).data.numpy().tolist()

        tp = tp / len(dsl.dataset)
        tn = tn / len(dsl.dataset)
        fp = fp / len(dsl.dataset)
        fn = fn / len(dsl.dataset)

        logger.add_scalar('{}/acc'.format(name), (tp + tn), global_step)
        logger.add_scalar('{}/fp'.format(name), (fp), global_step)
        logger.add_scalar('{}/fn'.format(name), (fn), global_step)

        print(name, ':', 'acc', tp + tn, 'fp', fp, 'fn', fn)

    test(test_loader, 'test')
    for ep in range(epochs):
        print('training: ep', ep + 1)
        train()
        test(test_loader, 'test')
    test(validation_loader, 'validation')
    torch.save(net.state_dict(), os.path.join(logdir, 'trained.pt'))
    logger.flush()


def worker(e, seed, opt, epochs, batch_sz, dim,
           lr, wdecay, nesterov,
           cl_type, rnn_steps, train_dataset, test_dataset, validation_dataset,
           momentum, eps,
           use_cuda, log_dir, grad_clip):
    dtime = str(datetime.now())[:19].replace(':', '-')
    log_subdir = '{},{},{},{},{},{},{},{},{},{},{}'.format(dtime, seed, opt, epochs, batch_sz, dim,
                                                           lr, wdecay, nesterov,
                                                           cl_type, rnn_steps)
    try:
        run_experiment(train_dataset, test_dataset, validation_dataset, batch_sz, e,
                       epochs, rnn_steps, dim, cl_type,
                       opt, lr, momentum, wdecay, eps, nesterov,
                       seed, use_cuda, os.path.join(log_dir, log_subdir), grad_clip)
    except Exception as e:
        print(e)


def list_worker(device, works, seed, opt, epochs, batch_sz,
                lr, wdecay, nesterov, train_dataset, test_dataset, validation_dataset,
                momentum, eps, log_dir, grad_clip):
    cuda = device

    for i, e in works:
        dim, cl_type, rnn_steps = e
        worker(e, seed, opt, epochs, batch_sz, dim,
               lr, wdecay, nesterov,
               cl_type, rnn_steps, train_dataset, test_dataset, validation_dataset,
               momentum, eps,
               cuda, log_dir, grad_clip)


def main():
    data_path = r'data path'

    log_dir = r'logs path'

    batch_sz = 5
    epochs = 15
    rnn_steps = 30
    cl_type = 1
    dim = 16
    seed = 42
    n_vars = 10
    n_ops = 55

    use_cuda = True

    data_debug = True

    print('loading train data')
    train_dataset = TreeFormulasDataset(os.path.join(data_path, 'train.txt'), n_ops, n_vars, data_debug)
    print('loading test data')
    test_dataset = TreeFormulasDataset(os.path.join(data_path, 'test.txt'), n_ops, n_vars, data_debug)
    print('loading validation data')
    validation_dataset = TreeFormulasDataset(os.path.join(data_path, 'validation.txt'), n_ops, n_vars, data_debug)

    experiment_target = 'steps_dim_cl'

    dim_options = [1, 2, 4, 8, 16, 32, 64]
    cl_options = [1, 2, 3, 5, 6, 7]
    rnn_steps_options = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]

    experiments = itertools.product(dim_options, cl_options, rnn_steps_options)

    opt = 'adam'
    lr = 0.005 if opt == 'sgd' else 0.0005
    momentum = 0.75
    nesterov = False
    wdecay = 1e-4
    eps = 1e-8
    grad_clip = 0.65

    n_gpus = 2

    gpus = ['cuda:{}'.format(i) for i in range(n_gpus)]

    works = []

    skip_e = None

    for i, e in enumerate(experiments):
        if skip_e is not None:
            if e == skip_e:
                skip_e = None
            else:
                continue
        works.append((i, e))

    works_by_gpus = [(gpus[i], works[i::3]) for i in range(len(gpus))]

    wlist=[delayed(list_worker)(gpu, wks, seed, opt, epochs, batch_sz,
                    lr, wdecay, nesterov, train_dataset, test_dataset, validation_dataset,
                    momentum, eps, log_dir, grad_clip) for gpu,wks in works_by_gpus]

    Parallel(len(wlist),'threading')(wlist)

    # for gpu, wks in works_by_gpus:
    #     list_worker(gpu, wks, seed, opt, epochs, batch_sz,
    #                 lr, wdecay, nesterov, train_dataset, test_dataset, validation_dataset,
    #                 momentum, eps, log_dir, grad_clip)


if __name__ == '__main__':
    main()
