from generate_tree.generator2 import *


def reduce(a: Form):
    if isinstance(a, NegForm) and isinstance(a.value, NegForm):
        return a.value.value
    else:
        return a


def close_open_branches(brs):
    sbrs = sorted(brs, key=len)
    res_lst = []
    for literals in sbrs:
        covered = [x for x in literals if x in res_lst]
        if len(covered) > 0:
            continue
        c1 = [x for x in literals if reduce(NegForm(x)) not in res_lst]
        c2 = [x for x in literals if reduce(NegForm(x)) in res_lst]
        if len(c1) > 0:
            na = random.choice(c1)
            res_lst.append(na)
        elif len(c2) > 0:
            na = random.choice(c2)
            res_lst.append(na)
            break

    return res_lst  # list({random.choice(x) for x in brs})


def generate_dataset(alg, N, bn, vs, s, ops, remove_dn):
    sats = set()
    unstas = set()

    while len(sats) < N // 2:
        print('=================================================================================')
        if alg == 1:
            ag = generate_branches(bn, s)
        else:
            t = makeubtree(ops)
            ag = node2expression_tree(t)
        # print(ag.tolist())
        ag.init_operations()
        ag.assign_leafs_with_vars(vs)
        print('generated formula:', ag)
        # print(f.prefixstr())
        # print('infix form:', f)
        if remove_dn:
            ag.remove_double_negations()
            print('removed double negations')
            print(ag)
        f = ag.ToFormula()
        print('prefix form:', f.prefixstr())
        # print('infix form:', f)
        open_branches = a_tableux([f])
        # print('open branches:', open_branches)
        opbr = simplify(open_branches)
        print('minimized open branches:', opbr)
        if len(open_branches) == 0:
            print('UNSAT')
            continue
        print('SAT')

        if f.prefixstr() in sats:
            print('already generated')
            continue

        contras = close_open_branches(opbr)
        print('contras:', contras)
        neg_sel_br = neg_con(contras)
        closed_f = ConForm(f, neg_sel_br)
        print('closed formula:', closed_f.prefixstr())
        open_branches = a_tableux([closed_f])
        print('branches:', open_branches)
        if len(open_branches) > 0:
            print('alarm')
        sats.add(f.prefixstr())
        unstas.add(closed_f.prefixstr())

    return sats, unstas


def generate_data_file(filename, n_branches, n_vars, tests, trains, validation):
    sats, unsats = generate_dataset(alg=1, N=tests + trains + validation, bn=n_branches, vs=n_vars, s=0.2, ops=140,
                                    remove_dn=True)
    sats = list(sats)
    unsats = list(unsats)
    datasets = {}
    for lbl, sz in zip(['train', 'test', 'validation'], [trains, tests, validation]):
        datasets[lbl] = []
        print(lbl, sz)
        for _ in range(0, sz, 2):
            s = random.choice(sats)
            datasets[lbl].append((1, s))
            sats.remove(s)
            us = random.choice(unsats)
            datasets[lbl].append((0, us))
            unsats.remove(us)
    for lbl in ['train', 'test', 'validation']:
        with open(filename.format(lbl), 'w') as fl:
            for l, s in datasets[lbl]:
                fl.write('{} {}\n'.format(l, s))
    print('done')


generate_data_file(r"C:\Users\Alex\Dropbox\институт\статья генерация примеров выполнимости\data\2\{}.txt", 15,
                   10, 10000, 50000, 10000)
