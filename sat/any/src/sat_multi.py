import numpy as np

mx = [[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

mx = np.asarray(mx)

mx_reduced = [[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]

mx_reduced = np.asarray(mx_reduced)

neg_ops = np.asarray([0, 0, 1, 0, 0, 0, 0, 1, 1])
con_ops = np.asarray([0, 1, 0, 0, 1, 1, 0, 0, 0])

solution = [0.001, 0.003, 0.006, 0.98, 0.002]

a = [0] * mx_reduced.shape[0]


def cond_neg(h, n):
    return (1 - h) * n + h * (1 - n)

fl=['zadeh','prababilistic'][1]


def eval_ops(g, m, con):
    if fl=='zadeh':
        conjunctions = np.min(np.maximum(1 - m, g), 1)
        disjunctions = np.max(g, 1)
    elif fl=='prababilistic':
        conjunctions = np.prod(np.maximum(1 - m, g), 1)
        disjunctions = 1-np.prod(1-g, 1)

    r = con * conjunctions + (1 - con)*disjunctions
    return r


def evaluation_step(mx, a, v, neg_ops, con_ops):
    g = mx * np.concatenate([a, v]).reshape([1, -1])
    h = eval_ops(g, mx, con_ops)
    r = cond_neg(h, neg_ops)
    return r


a=np.asarray(a)
for i in range(mx_reduced.shape[0]):
    a1 = evaluation_step(mx_reduced, a, solution, neg_ops, con_ops)
    print(list(a1))
    # if np.linalg.norm(a-a1)<0.01:
    #     break
    a=a1
    # if a[0]==1:
    #     break
