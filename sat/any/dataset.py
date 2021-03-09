from logic import prop
import numpy as np


def simple_read(fname):
    with open(fname, 'r') as f:
        ls = [x.rstrip('\n').split(' ') for x in f.readlines()]
    ls = [(int(x[0]), prop.Form.parse_prefix(x[1])) for x in ls]

    return ls


class graph:
    def __init__(self):
        self.vertices_names = []
        self.connections = []


class cntr:
    def __init__(self):
        self.c = 0

    def inc(self):
        self.c += 1


def formula2graph(f, opc=None, g=None):
    if g is None:
        g = graph()
    if opc is None:
        opc = cntr()
    if isinstance(f, prop.AtomForm):
        if f.name not in g.vertices_names:
            g.vertices_names.append(f.name)
        return [f.name], g
    elif isinstance(f, prop.NegForm):
        new_name = 'N{:0>3d}'.format(opc.c)
        opc.inc()
        g.vertices_names.append(new_name)
        childs, g = formula2graph(f.value, opc, g)
        g.connections.append((new_name, childs[0]))
        return [new_name], g
    elif isinstance(f, prop.ImpForm):
        ch = 'I'
        f1 = prop.DisForm(prop.NegForm(f.a), f.b)
        return formula2graph(f1, opc, g)
    elif isinstance(f, prop.BinForm):
        ch = 'C' if isinstance(f, prop.ConForm) else 'D'
        new_name = '{}{:0>3d}'.format(ch, opc.c)
        opc.inc()
        g.vertices_names.append(new_name)
        childs1, g = formula2graph(f.a, opc, g)
        childs2, g = formula2graph(f.b, opc, g)
        for c in childs1 + childs2:
            g.connections.append((new_name, c))
        return [new_name], g
    else:
        print()


def optimize_graph(vs, es):
    vs = vs.copy()
    es = es.copy()
    c = True
    for _ in range(len(vs)):

        ch = False
        for a, b in es:
            sims = []
            if a[0] == 'N':
                sims = {c for c, d in es if c[0] == a[0] and d == b and c != a}
            elif a[0] in ['C', 'I', 'D']:
                chs = {d for c, d in es if c == a}
                sims = {v for v in vs if v != a and all([((v, ch) in es) for ch in chs])}
            for c in sims:
                ch = True
                if c in vs:
                    vs.pop(vs.index(c))

                for g, h in [x for x in es if x[1] == c]:
                    es[es.index((g, c))] = (g, a)
                for g, h in [x for x in es if x[0] == c]:
                    es[es.index((c, h))] = (a, h)
        if not ch:
            break

    es = list(set(es))

    return vs, es


def formula2matrix(f, align_ops=70, n_vars=10):
    # f=prop.Form.parse_prefix('DDCv1Nv2Nv2Cv1Nv2')

    _, g = formula2graph(f)

    ops = [x for x in g.vertices_names if x[0] != 'v']
    vars = [x for x in g.vertices_names if x[0] == 'v']
    g.vertices_names = ops + vars
    a = len(g.vertices_names)

    v, c = optimize_graph(g.vertices_names, g.connections)

    g.vertices_names = v
    g.connections = c

    b = len(g.vertices_names)

    # print(a,b)

    ops = [x for x in g.vertices_names if x[0] != 'v']
    vars = [x for x in g.vertices_names if x[0] == 'v']
    mx = np.zeros([align_ops,
                   align_ops + n_vars
                   ], np.int8)
    for a, b in g.connections:
        if b in vars:
            mx[g.vertices_names.index(a), g.vertices_names.index(b) - len(ops) + align_ops] = 1
        else:
            mx[g.vertices_names.index(a), g.vertices_names.index(b)] = 1

    negops = np.zeros(align_ops, dtype=np.int8)
    conops = np.zeros(align_ops, dtype=np.int8)
    for i, o in enumerate(ops):
        if o[0] == 'N':
            negops[i] = 1
        elif o[0] == 'C':
            conops[i] = 1

    return mx, negops, conops


from torch.utils.data import Dataset

from joblib import delayed, Parallel


def f2matr_worker(i, matrices, x, vars, max_ops):
    matrices[i] = formula2matrix(x, max_ops, vars)


class TreeFormulasDataset(Dataset):
    def __init__(self, fname, max_ops=70, vars=10, debug=True):
        super().__init__()

        d = simple_read(fname)
        if debug:
            d = d[:1000]
        self.labels = np.asarray([x[0] for x in d])
        matrices = [None] * len(d)

        jobs = [delayed(f2matr_worker)(i, matrices, x[1], vars, max_ops) for i, x in enumerate(d)]
        Parallel(8, 'threading', 1)(jobs)
        #
        # [formula2matrix() for x in d]
        self.matrices = []
        self.negops = []
        self.conops = []
        for m, n, c in matrices:
            self.matrices.append(m)
            self.negops.append(n)
            self.conops.append(c)
        self.matrices = np.asarray(self.matrices)
        self.negops = np.asarray(self.negops)
        self.conops = np.asarray(self.conops)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {'label': self.labels[item],
                'matrix': self.matrices[item],
                'negops': self.negops[item],
                'conops': self.conops[item]}
