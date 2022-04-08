import matplotlib.pyplot as plt
import matplotlib.colors
from io import BytesIO
import cv2
from PIL import Image

def plot_dynamics(formulas, assignments, predicted_lbl, labels):
    imgs = [None] * len(formulas)

    plt.interactive(False)
    for r, fl in enumerate(formulas):

        plt.gcf().clear()
        a = assignments[r]
        pred = predicted_lbl[r]
        lab = labels[r]

        a = a.transpose()
        x_labels = ["t={}".format(t) for t in range(0, a.shape[1])]
        y_labels = ["x{}".format(t) for t in range(1, a.shape[0] + 1)] + ['', 'yp']

        cells = np.concatenate([a, [[0] * len(pred)], [pred]], 0)

        fig, ax = plt.subplots()
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
        # fig.tight_layout()
        #plt.show()
        with BytesIO() as buf:
            plt.savefig(buf, format='png')
            buf.seek(0)
            buf = buf.read(-1)
            img = cv2.imdecode(np.fromstring(buf, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs[r] = img

    return imgs


from dataset import TreeFormulasDataset
import os

data_path = '/mnt/nvme102/alex/sat/any/data'

log_dir = '/mnt/nvme102/alex/sat/any/logs2022'

from torch.utils.data import DataLoader

batch_sz = 250
epochs = 30
seed = 42
n_vars = 10
n_ops = 55
import numpy as np

data_debug = True
rnn_steps = 20
from nnet import SimpleTreeSAT

print('loading test data')
test_dataset = TreeFormulasDataset(os.path.join(data_path, 'test.txt'), n_ops, n_vars, data_debug)

test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)

net = SimpleTreeSAT(64, classifier_type=5)

for batch in test_loader:
    tracing_examples = 30
    res, tracing = net(mxs=batch['matrix'],
                       cons=batch['conops'],
                       negs=batch['negops'], rnn_steps=rnn_steps, trace=tracing_examples)

    trace_formulas = batch['formula'][:tracing_examples]

    trace_assignments = np.asarray([t[0] for t in tracing])
    trace_assignments = trace_assignments.transpose([1, 0, 2])

    trace_predictions = np.asarray([t[1] for t in tracing])
    trace_predictions = trace_predictions.transpose([1, 0])
    trace_labels = batch['label'].numpy()[:tracing_examples]
    imgs = plot_dynamics(trace_formulas, trace_assignments, trace_predictions, trace_labels)

print()
