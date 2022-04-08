import argparse

import numpy as np


def cnf2mx_any(cnf, nvars):
    nc = len(cnf)
    M = np.zeros([1 + nc + nvars, 1 + nc + nvars + nvars], np.uint8)
    N = np.zeros([1 + nc + nvars], np.uint8)
    C = np.zeros([1 + nc + nvars], np.uint8)

    C[0] = 1
    for i in range(1, nc + 1):
        M[0, i] = 1

    for i in range(nc + 1, nc + 1 + nvars):
        N[i] = 1
        M[i, i + nvars] = 1

    for i, clause in enumerate(cnf):
        for j, l in enumerate(clause):
            if l > 0:
                M[1 + i, 1 + nc + nvars + l - 1] = 1
            else:
                M[1 + i, 1 + nc - l - 1] = 1
    return M.astype(np.float), N.astype(np.float), C.astype(np.float)


def main(args):
    print('solving sat with neural network')

    cnf = []

    with open(args.formula_file, 'r') as ff:
        cnf = [[int(x) for x in l.rstrip('\n').split('\t')][:-1] for i, l in enumerate(ff) if i > 0]
    varname0 = 1
    varnamen = 0
    for clause in cnf:
        for l in clause:
            varnamen = max([varnamen, l, -l])
    print('numb. of variables: {:d}'.format(varnamen))
    print('numb. of clauses:   {:d}'.format(len(cnf)))

    use_nnet = 'any'

    #region load neural network
    if use_nnet=='any':
        model_weights='/mnt/nvme102/alex/sat/any/logs'
    else:
        print('E')

    #region init SAT matrix form

    print('init vector-matrix form')
    if use_nnet == 'any':

        M, N, C = cnf2mx_any(cnf, varnamen)


    else:
        print('E')

    #endregion

    print('matrix form init ok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('formula_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()
    main(args)
