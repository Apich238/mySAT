import math
from time import time
import random
from joblib import Parallel, delayed

import numpy as np

from logic.prop import AtomForm, ImpForm, ConForm, DisForm, NegForm, CNForm
from graph import graph


class node:
    def __init__(self, arity=2):
        self.childs = [None for _ in range(arity)]
        self.op = ''

    def random_insert_node(self):
        c = np.random.choice(range(len(self.childs)), 1)[0]
        if self.childs[c] is None:
            self.childs[c] = random_node(allow_unary=(len(self.childs) > 1))
        else:
            self.childs[c].random_insert_node()

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return 'node ch={} op={}'.format(len(self.childs), self.op)

    def make_ops(self):
        if len(self.childs) == 1:
            self.op = 'N'
        else:
            self.op = np.random.choice(['C', 'D', 'I'])
        for c in self.childs:
            if c is not None:
                c.make_ops()

    def try_to_push_var(self, v):

        perm = [i for i in np.random.permutation(np.arange(len(self.childs)))]

        for i in perm:
            if self.childs[i] is None:
                self.childs[i] = v
                return True
            elif isinstance(self.childs[i], str):
                continue
            elif isinstance(self.childs[i], node):
                if self.childs[i].try_to_push_var(v):
                    return True
        return False


def _printNode(anode, t=0):
    print('{}{}\n'.format('\t' * t, str(anode)), end='')
    if isinstance(anode, node):
        for c in anode.childs:
            _printNode(c, t + 1)


def random_node(allow_unary=True):
    if allow_unary:
        return node(np.random.choice([1, 2], 1, p=[0.2, 0.8])[0])
    else:
        return node(2)


def generate_unbin_tree(ops):
    root = random_node()

    n = ops - 1
    while (n > 0):
        root.random_insert_node()
        n -= 1

    return root


def make_vars(cnt):
    return ['p{}'.format(i) for i in range(cnt)]


def insert_vars(vars, root):
    f = True
    i = 0
    while f:
        v = vars[i % len(vars)]
        f = root.try_to_push_var(v)
        i += 1


def make_formula(node):
    if isinstance(node, str):
        return AtomForm(node)
    elif node.op == 'D':
        return DisForm(make_formula(node.childs[0]), make_formula(node.childs[1]))
    elif node.op == 'C':
        return ConForm(make_formula(node.childs[0]), make_formula(node.childs[1]))
    elif node.op == 'I':
        return ImpForm(make_formula(node.childs[0]), make_formula(node.childs[1]))
    elif node.op == 'N':
        return NegForm(make_formula(node.childs[0]))


def generateRandomFormula(nVars, nOps=None):
    if nOps is None:
        nOps = int(np.random.uniform(2 * nVars, nVars * 4))
    t = generate_unbin_tree(nOps)
    t.make_ops()
    vs = make_vars(nVars)
    insert_vars(vs, t)
    f = make_formula(t)
    return f


def generateRandomCNFt(nVars, clause_max=-1, clause_min=1, exact_clauses=None):
    if exact_clauses is not None:
        num = exact_clauses
    else:
        min_disjuncts = clause_min
        max_disjuncts = 2 ** nVars
        if clause_max > 0:
            max_disjuncts = min(clause_max, max_disjuncts)
        num = int(round(np.random.uniform(min_disjuncts, max_disjuncts)))
    s = set()
    for _ in range(num):
        l = []
        for i in range(1, 1 + nVars):
            c = np.random.choice((0, 0, 0, 1, 2))
            if c == 1:
                l.append(i)
            elif c == 2:
                l.append(-i)
        if len(l) > 0:
            s.add(tuple(l))
    return list(s)


def CNFt2CNF(s):
    ll = [[AtomForm('p{}'.format(str(n))) if n > 0 else NegForm(AtomForm('p{}'.format(str(-n))))
           for n in t] for t in s]
    f = CNForm(ll)
    if minimize:
        pass
    return f


def toCNF(f):
    pass


def CNF_open_branches(f: CNForm):
    def atree_starter(cons):
        branch = cons
        return atree_rec(branch, set())

    def extend_liters(a, ls: set):
        nl = ls.copy()

        inv = None
        if isinstance(a, NegForm):
            inv = a.value
        else:
            inv = NegForm(a)
        if inv in nl:
            return None

        nl.add(a)
        return nl

    def atree_rec(branch, liters):
        if len(branch) == 0:
            return [list(liters)]
        else:
            res = []
            b = branch.copy()
            dis = b.pop(0)
            for v in dis:
                nl = extend_liters(v, liters)
                if nl is not None:
                    r = atree_rec(b, nl)
                    if r is not None:
                        res.extend(r)
            return res

    open_branches = atree_starter(f.members)
    return open_branches


def CNFt_open_branches(f):
    def extend_liters(a, ls):
        nl = ls  # .copy()

        if -a in nl:
            return None
        elif a in nl:
            return nl
        else:
            return nl + (a,)

    def optimize_set(s0):
        if len(s0) < 2:
            return s0
        s = list(s0)
        s.sort(key=lambda l: len(l))
        i = 0
        while i < len(s):
            a = s[i]
            j = i + 1
            while j < len(s):
                b = s[j]
                if is_strong_subset(a, b):
                    s.pop(j)
                else:
                    j += 1
            i += 1
        return set(s)

    def atree_rec(branch, liters=()):
        if len(branch) == 0:
            return {tuple(sorted(liters, key=math.fabs))}
        else:
            res = set()
            b = branch.copy()
            dis = b.pop(0)
            for v in dis:
                nl = extend_liters(v, liters)
                if nl is None:
                    continue
                rb = sorted([tuple([m for m in x if m != -v]) for x in b if v not in x], key=len)
                r = atree_rec(rb, nl)
                if r is not None and len(r) > 0:
                    res.update(r)
                    # res.extend(r)
            # if len(res) > 5:
            #     res = optimize_set(res)
            return res

    open_branches = atree_rec(f)
    return optimize_set(open_branches)


def is_strong_subset(a, b):
    # a<b
    if len(a) >= len(b):
        return False
    for am in a:
        if not am in b:
            return False
    return True


def minimize_CNFt(f):
    f1 = f.copy()
    f = f1
    f.sort(key=lambda l: len(l))
    delta = True

    while delta and len(f) > 1:
        delta = False
        res = []
        for c in f:
            if c in res:
                continue
            else:
                fl = True
                for d in res:
                    fl = fl and not is_strong_subset(d, c)
                if fl:
                    res.append(c)
                else:
                    delta = True

        res1 = []
        for i in range(len(res)):
            a = res[i]
            to_add = a
            for j in range(i + 1, len(res)):
                b = res[j]
                if len(b) > len(a):
                    break
                x1 = np.asarray(a)
                x2 = np.asarray(b)
                e1 = x1 == x2
                e2 = x1 == -x2
                e3 = np.logical_xor(e1, e2)
                if np.all(e3) and np.count_nonzero(e2) == 1:
                    k = np.argmax(e2)
                    to_add = tuple([a[i] for i in range(len(a)) if i != k])
                    delta = True

            res1.append(to_add)

        res1.sort(key=lambda l: len(l))
        f = res1
        # print('simplified:\n',f)
    return f


def CNFtto3SAT(f):
    if all([len(x) <= 3 for x in f]):
        return f

    res = [x for x in f if len(x) <= 3]

    tomin = [x for x in f if len(x) > 3]

    nextvar = np.abs(np.concatenate(f)).max() + 1

    while len(tomin) > 0:
        to_reduce = list(tomin.pop(0))
        appends = []
        while len(to_reduce) > 3:
            H = to_reduce[:-2]
            c = to_reduce[-2]
            d = to_reduce[-1]
            e = nextvar
            nextvar += 1
            appends.extend([(-e, c, d), (e, -c), (e, -d)])
            to_reduce = H + [e]
        appends.append(to_reduce)
        res.extend(appends)

    res.sort(key=len)
    return res


# for _ in range(30):
#     f=generateRandomFormula(nVars=5,nOps=20)
#     print(f)
#     print(f.prefixstr())

resdict = {}


def run_test(nVars, samples=1000):
    nV = nVars
    cc = 0
    co = 0
    tgen = 0.
    tsimp = 0.
    tsat3 = 0.
    tatables = 0.
    tatablessat3 = 0.
    ttotal = time()
    for i in range(samples):
        t = time()
        f = generateRandomCNFt(nV)
        tgen += time() - t
        print('f {}==============================================================:\n'.format(i))  # , f)
        t = time()
        min_f = minimize_CNFt(f)
        tsimp += time() - t
        print('minimal:\n', min_f)
        t = time()
        obranches = CNFt_open_branches(min_f)
        tatables += time() - t
        if len(obranches) == 0:
            print('closed formula!!!')
            cc += 1
        else:
            print('minimal branches ({}):\n'.format(len(obranches)), obranches)
            co += 1
        t = time()
        sat3 = CNFtto3SAT(min_f)
        tsat3 += time() - t
        print('3sat:\n', sat3)
        t = time()
        obranches = CNFt_open_branches(sat3)
        tatablessat3 += time() - t
        if len(obranches) == 0:
            print('closed formula!!!')
        else:
            print('minimal branches ({}):\n'.format(len(obranches)), obranches)

    dt = (time() - ttotal)
    resdict[nVars] = {
        'avgtime': dt / samples,
        'generation': tgen / samples,
        'simplification': tsimp / samples,
        'converting to SAT3': tsat3 / samples,
        'atabeaux for sat': tatables / samples,
        'atabeaux for sat3': tatablessat3 / samples,
        'satisfiable': co / (cc + co),
        'unsatisfiable': cc / (cc + co),
    }

    # print(
    #    '{: >4d} variables (average {:.5f} s. per example): unsat:{:.2f}, sat:{:.2f}'.format(nV, dt, cc / (cc + co), ))


# g=graph()
# g.add_edge('a','b')
# g.print()
# print('----------------')
# g.add_edge('a','c')
# g.print()
# print('----------------')
# g.add_edge('b','c')
# g.print()
# print('----------------')
# g.add_edge('b','c')
# g.print()
# print('----------------')


def make_graph_tree(f):
    g = graph()

    for i, c in enumerate(f):
        dname1 = 'd{}'.format(i)
        g.add_edge('g', dname1)
        g.add_edge('g', 'c')
        for v in c:
            if v < 0:
                vname1 = 'p{}'.format(-v)
                g.add_edge(vname1, 'v{}'.format(-v))
                vname2 = 'n{}'.format(-v)
                g.add_edge(vname1, vname2)
                vname = vname2
            else:
                vname = 'v{}'.format(int(math.fabs(v)))
            g.add_edge(dname1, vname)

    return g


def make_graph_fc1(f):
    pass


def make_graph_2partial(f):
    g = graph()
    for i, c in enumerate(f):
        for v in c:
            g.add_edge('c{}'.format(i), 'v{}'.format(int(math.fabs(v))), int(v / math.fabs(v)))
    return g


def test():
    for v in range(4, 10):
        run_test(v, 50)  # int(2 ** (6 * 12 / v)))

    for v in resdict:
        print('{}:'.format(v))
        for k in resdict[v]:
            print('\t{}:\t{:.5f}'.format(k, resdict[v][k]))

    f = generateRandomCNFt(3)
    print(f)
    g2p = make_graph_2partial(f)
    g2p.print()
    mx, lbls = g2p.getAdjMatrix()
    print(lbls)
    print(mx)

    gtree = make_graph_tree(f)
    gtree.print()
    mx, lbls = gtree.getAdjMatrix()
    print(lbls)
    print(mx)


def SolveSAT(f):
    min_f = minimize_CNFt(f)
    if len(min_f) == 0:
        return False
    obranches = CNFt_open_branches(min_f)
    return len(obranches) > 0


def effective_Vars(f):
    return len(set([abs(a) for a in sum([list(x) for x in f], [])]))


import networkx as nx
from networkx.algorithms import bipartite


def cnf2graph(f):
    g = nx.Graph()
    vs = set()
    cs = set()
    for i, cl in enumerate(f):
        c = 'C{}'.format(i)
        g.add_node(c, bipartite=1)
        cs.add(c)
        for j, vr in enumerate(cl):
            v = 'V{}'.format(abs(vr))
            vs.add(v)
            g.add_node(v, bipartite=0)
            g.add_edge(v, c, weight=1 if vr > 0 else -1)
    return g, vs, cs


def normalizeCNF(f):
    # print(f)
    g, vs, cs = cnf2graph(f)
    # разделяем по компонентам связности, сортируем их по убыванию количества вершин
    components = list(nx.connected_components(g))
    components = sorted(components, key=len, reverse=True)
    subgraphs = [g.subgraph(nds) for nds in components]

    # количество соседей каждой вершины
    neighboursV = {k: len(list(nx.neighbors(g, k))) for k in g.nodes() if k in vs}
    neighboursC = {k: len(list(nx.neighbors(g, k))) for k in g.nodes() if k in cs}

    # def make_key(sg):
    #     ndsV = tuple(sorted([neighboursV[nd] for nd in sg.nodes if nd in neighboursV]))
    #     ndsC = tuple(sorted([neighboursC[nd] for nd in sg.nodes if nd in neighboursC]))
    #     return (ndsC, ndsV)
    #
    # k = sorted([make_key(sg) for sg in subgraphs])

    new_f = []
    new_vars = []
    new_clauses = []
    for sg in subgraphs:
        locvs = sorted([n for n in sg.nodes if n in vs], key=lambda n: neighboursV[n])
        loccs = sorted([n for n in sg.nodes if n in cs], key=lambda n: neighboursC[n])
        new_vars.extend(locvs)
        new_clauses.extend(loccs)
    new_clauses = [int(c[1:]) for i, c in enumerate(new_clauses)]
    new_vars = [int(v[1:]) for i, v in enumerate(new_vars)]

    for i, old_cl in enumerate(new_clauses):
        ocl = f[old_cl]
        new_cl = []
        for v in ocl:
            nv = new_vars.index(abs(v)) + 1
            if v < 0:
                nv *= -1
            new_cl.append(nv)
        new_cl = tuple(sorted(new_cl, key=abs))
        new_f.append(new_cl)
    if not (nx.algorithms.isomorphism.is_isomorphic(g, cnf2graph(new_f)[0])):
        raise Exception('graps before and after is not isomorphic')
    return tuple(new_f)  # tuple(sorted(f))


def generate_dataset(nVars, count, max_clauses=-1, min_clauses=1):
    data = set()
    c0 = 0
    c1 = 0
    while len(data) < count:
        # выбираем количество переменных
        if isinstance(nVars, int):
            nV = nVars
        else:
            nV = random.choice(nVars)
        #генерируем формулу
        eff_v = 0
        while eff_v != nV:
            f = generateRandomCNFt(nV, max_clauses, min_clauses)
            eff_v = effective_Vars(f)
            print('nV={},effV={},clauses={}'.format(nV, eff_v, len(f)))
        #нормализуем её
        f1 = normalizeCNF(f)
        # print('f1={}'.format(f1))
        #определяем выполнимость
        l = SolveSAT(f)
        #добавляем в соответствующее множество
        print('sat: {};'.format(l), end='')
        sz0 = len(data)
        if l and c1 * 2 < count:
            data.add((f1, l))
            sz1 = len(data)
            c1 += 1 if sz0 < sz1 else 0
            print('  added;', end='')
        elif not l and c0 * 2 < count:
            data.add((f1, l))
            sz1 = len(data)
            c0 += 1 if sz0 < sz1 else 0
            print(' added;', end='')
        else:
            print(' not added;', end='')
        print('     sat:{: >8}, unsat:{: >8}'.format(c1, c0))
    data = list(data)
    return data


# generate_dataset(r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data\test.txt', 4, 10)


def save_dataset(data, save_pth):
    with open(save_pth, 'w', encoding='utf-8') as f:
        for d in data:
            f.write('{}\n'.format(d))




def split(data, test_sz):
    t0 = []
    t1 = []
    tr = []
    for f, l in data:
        if not l and len(t0) * 2 < test_sz:
            t0.append((f, 0))
        elif l and len(t1) * 2 < test_sz:
            t1.append((f, 1))
        else:
            tr.append((f, 1 if l else 0))
    return tr, t0 + t1


def gen_dataset_files(nVarsT, trainSZ, testSZ, nVarsV, valSZ, svdir, max_clauses=-1):
    import os
    data = generate_dataset(nVarsT, trainSZ + testSZ, max_clauses, min_clauses=3)
    trdata, tsdata = split(data, testSZ)
    valdata = generate_dataset(nVarsV, valSZ, max_clauses, min_clauses=3)
    st = str(nVarsT) if isinstance(nVarsT, int) else '({}-{})'.format(min(nVarsT), max(nVarsT))
    stv = str(nVarsV) if isinstance(nVarsV, int) else '({}-{})'.format(min(nVarsV), max(nVarsV))
    if not os.path.exists(svdir):
        os.makedirs(svdir)
    save_dataset(trdata, os.path.join(svdir, 'V{}Train.txt'.format(st)))
    save_dataset(tsdata, os.path.join(svdir, 'V{}Test.txt'.format(st)))
    save_dataset(valdata, os.path.join(svdir, 'V{}Valid.txt'.format(stv)))
    print('done')


if __name__ == '__main__':
    train_size = 90000
    test_size = 10000
    val_size = 10000
    nVarsT = range(3, 8)
    nVarsV = range(10, 12)
    max_clauses = 200
    fld = r'C:\Users\Alex\Dropbox\институт\диссертация\статья нейросеть выводимость\data35'

    gen_dataset_files(nVarsT, train_size, test_size, nVarsV, val_size, fld, max_clauses)

# d,l=read_dataset(r'C:\Users\Alex\Dropbox\институт\диссертация\конфа 2020\data\V6Train.txt')
