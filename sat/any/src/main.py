import os
from datetime import datetime
import itertools

from joblib import delayed, Parallel

from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset import TreeFormulasDataset
from nnet import SimpleTreeSAT

import matplotlib.pyplot as plt
import matplotlib.colors
from io import BytesIO
import cv2
from PIL import Image

matplotlib.use('agg')


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
        a = 0
        net.train()
        for i, batch in enumerate(train_loader):
            opt.zero_grad()
            res, _ = net(mxs=batch['matrix'].to(device),
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
            if i % 10 == 0 and i > 0:
                print('step', i, ':', 'acc', (tp + tn) / len(lbl), 'fp', fp / len(lbl), 'fn', fn / len(lbl), 'loss', l)

    def plot_dynamics(formulas, assignments, predicted_lbl, labels):
        imgs = [None] * len(formulas)

        plt.interactive(False)
        for r, fl in enumerate(formulas):

            # plt.gcf().clear()
            fig, ax = plt.subplots()
            a = assignments[r]
            pred = predicted_lbl[r]
            lab = labels[r]

            a = a.transpose()
            x_labels = ["t={}".format(t) for t in range(0, a.shape[1])]
            y_labels = ["x{}".format(t) for t in range(1, a.shape[0] + 1)] + ['', 'yp']

            cells = np.concatenate([a, [[0] * len(pred)], [pred]], 0)

            im = ax.imshow(cells, cmap=plt.get_cmap("gray"), aspect=0.5, vmin=0., vmax=1., )
            fig.set_figwidth(10)
            # fig.set_figheight(5)

            # We want to show all ticks...
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_yticks(np.arange(len(y_labels)))
            # ... and label them with the respective list entries
            ax.set_xticklabels(x_labels)
            ax.set_yticklabels(y_labels)

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            for i in range(len(x_labels)):
                for j in list(range(len(y_labels) - 2)) + [len(y_labels) - 1]:
                    text = ax.text(i, j, round(cells[j, i], 2),
                                   ha="center", va="center", color="black" if cells[j, i] > 0.5 else 'white')

            ax.set_title(('SAT' if lab else 'UNSAT') + ' ' + fl.replace('v', 'x'))

            # plt.savefig('plots/{}.png'.format(r))
            #
            # plt.gcf().clear()
            # plt.close()
            # img = cv2.cvtColor(cv2.imread('plots/{}.png'.format(r)), cv2.COLOR_BGR2RGB)

            bts = BytesIO()
            plt.savefig(bts, format='png', bbox_inches='tight')
            bts.seek(0)
            buff = bts.read(-1)

            img = cv2.cvtColor(cv2.imdecode(np.frombuffer(buff, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            bts.close()
            plt.gcf().clear()
            plt.close()
            # print(r, img.shape)
            imgs[r] = img
        a = 0
        return imgs

    def test(dsl: DataLoader, name):
        tp, tn, fp, fn = 0, 0, 0, 0
        net.eval()
        k = 0
        with torch.no_grad():
            for batch in dsl:
                tracing_examples = 20
                res, tracing = net(mxs=batch['matrix'].to(device),
                                   cons=batch['conops'].to(device),
                                   negs=batch['negops'].to(device), rnn_steps=rnn_steps, trace=tracing_examples)
                res = res.cpu() > 0.5
                lbl = batch['label'].to(torch.bool)

                tp += torch.sum(res * lbl).data.numpy().tolist()
                tn += torch.sum((~res) * (~lbl)).data.numpy().tolist()
                fp += torch.sum(res * (~lbl)).data.numpy().tolist()
                fn += torch.sum((~res) * lbl).data.numpy().tolist()
                if k == 0:
                    trace_formulas = batch['formula'][:tracing_examples]

                    trace_assignments = np.asarray([t[0] for t in tracing])
                    trace_assignments = trace_assignments.transpose([1, 0, 2])

                    trace_predictions = np.asarray([t[1] for t in tracing])
                    trace_predictions = trace_predictions.transpose([1, 0])
                    trace_labels = batch['label'].numpy()[:tracing_examples]
                    imgs = plot_dynamics(trace_formulas, trace_assignments, trace_predictions, trace_labels)

                    for k, img in enumerate(imgs):
                        logger.add_image('{}/{}'.format(name, k), img, global_step, dataformats='HWC')
                    k = 1
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
                                                           lr, wdecay, nesterov, cl_type, rnn_steps)
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
    data_path = '/mnt/nvme102/alex/sat/any/data'

    log_dir = '/mnt/nvme102/alex/sat/any/logs2022/only_sat'

    batch_sz = 250
    epochs = 30
    seed = 42
    n_vars = 10
    n_ops = 55

    use_cuda = True

    data_debug = False

    print('loading train data')
    train_dataset = TreeFormulasDataset(os.path.join(data_path, 'train.txt'), n_ops, n_vars, data_debug,only_sat=True)
    print('loading test data')
    test_dataset = TreeFormulasDataset(os.path.join(data_path, 'test.txt'), n_ops, n_vars, data_debug)
    print('loading validation data')
    validation_dataset = TreeFormulasDataset(os.path.join(data_path, 'validation.txt'), n_ops, n_vars, data_debug)

    experiment_target = 'steps_dim_cl'

    dim_options = [  # 1, 2, 4, 8, 16, 32,
        16]
    cl_options = [  # 1, 2, 3,
        5,
        6,
        7
    ]  # 5=min, 6=prod,7=luk
    rnn_steps_options = [  # 0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50
        10]

    experiments = itertools.product(dim_options, cl_options, rnn_steps_options)

    opt = 'adam'
    lr = 0.005 if opt == 'sgd' else 0.0005
    momentum = 0.75
    nesterov = False
    wdecay = 1e-4
    eps = 1e-8
    grad_clip = 0.2

    n_gpus = 1

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

    works_by_gpus = [(gpus[i], works[i::len(gpus)]) for i in range(len(gpus))]
    wlist = [delayed(list_worker)(gpu, wks, seed, opt, epochs, batch_sz,
                                  lr, wdecay, nesterov, train_dataset, test_dataset, validation_dataset,
                                  momentum, eps, log_dir, grad_clip) for gpu, wks in works_by_gpus]

    Parallel(len(wlist), 'threading')(wlist)

    # for gpu, wks in works_by_gpus:
    #     list_worker(gpu, wks, seed, opt, epochs, batch_sz,
    #                 lr, wdecay, nesterov, train_dataset, test_dataset, validation_dataset,
    #                 momentum, eps, log_dir, grad_clip)


if __name__ == '__main__':
    main()
