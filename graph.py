import numpy as np


class graph:
    def __init__(self):
        self.vertices=set()
        self.edges=set()

    def add_edge(self,a,b,w=1,two_way=True,w_inv=None):
        self.vertices.add(a)
        self.vertices.add(b)
        self.edges.add((a,b,w))
        if two_way:
            if w_inv is None:
                w_inv=w
            self.edges.add((b,a,w_inv))

    def print(self):
        print(self.vertices)
        for a,b,w in self.edges:
            print('{} -> {} : {}'.format(a,b,w))

    def getAdjMatrix(self):
        lbls=list(self.vertices)
        lbls.sort()
        mx=np.zeros([len(lbls),len(lbls)],dtype=float)
        for a,b,w in self.edges:
            mx[lbls.index(a),lbls.index(b)]=w
        return mx,lbls
