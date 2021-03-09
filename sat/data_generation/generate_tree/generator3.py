# DEEP LEARNING FOR SYMBOLIC MATHEMATICS

def memoize(f):
    cache = {}

    def decorate(*args):
        if args in cache:
            return cache[args]
        else:
            cache[args] = f(*args)
            return cache[args]

    return decorate


@memoize
def D(e, n):
    if e == 0:
        return 0
    elif n == 0:
        return 1
    else:
        return D(e - 1, n) + D(e, n - 1) + D(e + 1, n - 1)


def PL(e, n, a, k):
    if a == 1:
        return D(e - k, n - 1) / D(e, n)
    elif a == 2:
        return D(e - k + 1, n - 1) / D(e, n)
    else:
        return 0.


def PK(e, n, k):
    return D(e - k + 1, n - 1) / D(e, n)


import random


def sampleL(e, n):
    while True:
        s = random.uniform(0., 1.)
        for a in [1, 2]:
            for k in range(0, e, 1):
                prob = PL(e, n, 1, k)
                if s < prob:
                    return a, k
                s -= prob


def sampleK(e, n):
    while True:
        s = random.uniform(0., 1.)
        for k in range(0, e, 1):
            prob = PK(e, n, k)
            if s < prob:
                return k
            s -= prob


class node:
    def __init__(self):
        '''
        none=empty node
        []=leaf
        '''
        self.childs = None
        self.lbl = 'empty'

    def set_arity(self, ar):
        for _ in range(ar):
            self.childs.append(node())

    def insert_first(self, ar, lbl):
        if self.childs is None:
            self.childs = [node() for _ in range(ar)]
            self.lbl = lbl
            return self
        else:
            for c in self.childs:
                r = c.insert_first(ar, lbl)
                if r is not None:
                    return r
            return None

    def print(self, t=0, sp=1):
        print(' ' * t * sp + self.lbl)
        if self.childs is not None:
            for c in self.childs:
                c.print(t + 1, sp)

    def has_empty_nodes(self):
        if self.childs is None:
            return True
        for c in self.childs:
            if c.has_empty_nodes():
                return True
        return False


def makeubtree(n):
    e = 1
    t = node()
    while n > 0:
        a, k = sampleL(e, n)
        for _ in range(k):
            t.insert_first(0, None)  # 'leaf')
        if a == 1:
            op = 'neg'
            t.insert_first(1, op)
            e = e - k
        else:
            op = None  # random.choice(['con', 'dis'])
            t.insert_first(2, op)
            e = e - k + 1
        n = n - 1
    while t.has_empty_nodes():
        t.insert_first(0, 'leaf')
    return t


# for _ in range(10):
#     t = makeubtree(5)
#     print()
#     t.print(sp=2)

# назначение переменных
