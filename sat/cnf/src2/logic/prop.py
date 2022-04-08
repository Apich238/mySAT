from logic.myPEG import *


class Form:

    def __str__(self):
        return self.repr()

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return hash(self)==hash(other)

    def __hash__(self):
        return self.__repr__().__hash__()

    def prefixstr(self):
        pass

    @staticmethod
    def syn2ast(node: TNode):
        if len(node.childs) == 2 and node.childs[1].symbol == 'zom':
            sym = 'con' if node.symbol in ['propcon', 'estcon'] else 'dis'
            res = TNode(sym)
            res.add(Form.syn2ast(node.childs[0]))
            for sq in node.childs[1].childs:
                if sq.symbol == 'seq':
                    # for c in sq.childs:
                    #     res.add(syn2ast(c))
                    res.add(Form.syn2ast(sq.childs[1]))
            return res
        elif node.symbol == 'seq' and len(node.childs) == 2 and \
                isinstance(node.childs[0].symbol, Token) and \
                node.childs[0].symbol.type == 'neg':

            res = TNode(node.childs[0].symbol)
            res.add(Form.syn2ast(node.childs[1]))
            return res
        elif node.symbol == 'seq' and len(node.childs) == 3 and \
                isinstance(node.childs[0].symbol, Token) and \
                node.childs[0].symbol.type == 'ops':
            return Form.syn2ast(node.childs[1])
        elif node.symbol in ['propimp', 'estimp']:
            a = Form.syn2ast(node.childs[0])
            b = Form.syn2ast(node.childs[2])
            res = TNode('imp')
            res.add(a)
            res.add(b)
            return res
        elif node.symbol == 'estprop':
            res = TNode(node.childs[1].symbol)
            res.add(Form.syn2ast(node.childs[0]))
            res.add(node.childs[2])
            return res
        elif len(node.childs) > 0 and \
                isinstance(node.childs[0].symbol, Token) and \
                node.childs[0].symbol.type == 'sign':
            res = TNode(node.childs[0].symbol)
            for c in node.childs[1:]:
                res.add(Form.syn2ast(c))
            return res
        elif len(node.childs) == 1:
            return Form.syn2ast(node.childs[0])
        else:
            res = TNode(node.symbol)
            for c in node.childs:
                res.add(Form.syn2ast(c))
            return res

    @staticmethod
    def compile_ast(node):
        if isinstance(node.symbol, Token) and node.symbol.type == 'sign':
            return SignedForm(Form.compile_ast(node.childs[0]), node.symbol.value)
        elif isinstance(node.symbol, Token) and node.symbol.type == 'atom':
            return AtomForm(node.symbol.value)
        elif node.symbol == 'imp':
            return ImpForm(Form.compile_ast(node.childs[0]), Form.compile_ast(node.childs[1]))
        elif node.symbol == 'con':
            return ConForm([Form.compile_ast(c) for c in node.childs])
        elif node.symbol == 'dis':
            return DisForm([Form.compile_ast(c) for c in node.childs])
        elif isinstance(node.symbol, Token) and node.symbol.type == 'neg':
            return NegForm(Form.compile_ast(node.childs[0]))
        raise ValueError('Unknorn node type:'.format(node))

    @staticmethod
    def parse_formula(l: str):
        syntree = propPEG.Parse(l)
        ast = Form.syn2ast(syntree)
        res = Form.compile_ast(ast)
        return res

    @staticmethod
    def parse_prefix(l: str, i=0):
        import re
        if i >= len(l):
            raise IndexError()
        elif l[i] == 'C':
            a, j = Form.parse_prefix(l, i + 1)
            b, j = Form.parse_prefix(l, j)
            res = ConForm(a, b)
        elif l[i] == 'D':
            a, j = Form.parse_prefix(l, i + 1)
            b, j = Form.parse_prefix(l, j)
            res = DisForm(a, b)
        elif l[i] == 'I':
            a, j = Form.parse_prefix(l, i + 1)
            b, j = Form.parse_prefix(l, j)
            res = ImpForm(a, b)
        elif l[i] == 'N':
            a, j = Form.parse_prefix(l, i + 1)
            res = NegForm(value=a)
        else:  # variable
            ptrn = re.compile("p[0-9]+")
            reres = ptrn.match(l, pos=i)
            if reres is None:
                raise ValueError('incorrect string at pos={}: {}'.format(i, l))
            a = reres.group(0)

            res = AtomForm(a)
            j = reres.end(0)

        if i == 0:
            if j < len(l):
                raise ValueError('remain redundant symbols')
            return res
        else:
            return res, j


class AtomForm(Form):
    def __init__(self, name):
        super(Form, self).__init__()
        self.name = name

    def repr(self, br=False):
        return self.name

    def prefixstr(self):
        return self.name


class NegForm(Form):
    def __init__(self, value):
        super(Form, self).__init__()
        self.value = value

    def repr(self, br=False):
        return ('~{}' if isinstance(self.value, (AtomForm, NegForm)) else '~({})').format(self.value)

    def prefixstr(self):
        return 'N' + self.value.prefixstr()


class BinForm(Form):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b


class ConForm(BinForm):
    def __init__(self, a, b):
        super().__init__(a, b)

    def repr(self, br=False):
        ms = [self.a, self.b]
        return ('({})' if br else '{}').format(
            '&'.join([a.repr(br=(isinstance(a, (DisForm, ImpForm)))) for a in ms]))

    def prefixstr(self):
        return 'C' + self.a.prefixstr() + self.b.prefixstr()


class DisForm(BinForm):
    def __init__(self, a, b):
        super().__init__(a, b)

    def repr(self, br=False):
        ms = [self.a, self.b]
        return ('({})' if br else '{}').format('|'.join([
            a.repr(br=(isinstance(a, ImpForm)))
            for a in ms]))

    def prefixstr(self):
        return 'D' + self.a.prefixstr() + self.b.prefixstr()


class ImpForm(BinForm):
    def __init__(self, a: Form, b: Form):
        super().__init__(a, b)

    def repr(self, br=False):
        sa = ('({})' if isinstance(self.a, (ImpForm)) else '{}').format(self.a)
        sb = ('({})' if isinstance(self.b, (ImpForm)) else '{}').format(self.b)
        return ('({}=>{})' if br else '{}=>{}').format(sa, sb)

    def prefixstr(self):
        return 'I' + self.a.prefixstr() + self.b.prefixstr()


class SignedForm(Form):
    def __init__(self, expr, positive=True):
        self.positive = positive
        self.expr = expr

    def __repr__(self):
        return '{}{}'.format('+' if self.positive else '-', self.expr)


class CNForm(Form):
    def __init__(self, disjuncts: list):
        super().__init__()
        self.members = disjuncts.copy()

    def repr(self, br=False):
        return ('({})' if br else '{}').format(
            '&'.join(['({})'.format('|'.join([str(x) for x in d])) for d in self.members]))


propPEG = PEG('start',
              {
                  'atom': '[a-zA-Z][0-9a-zA-Z]*',
                  'ops': '\(',
                  'cls': '\)',
                  'neg': '~',
                  'disj': r'\|',
                  'conj': r'&',
                  'impl': r'=>',
                  'sign': r'\+|-',
              },
              {
                  'prop': sel('propimp', 'propdis'),
                  'propimp': ['propdis', 'impl', 'propdis'],
                  'propdis': ['propcon', zom(['disj', 'propcon'])],
                  'propcon': ['atomicprop', zom(['conj', 'atomicprop'])],
                  'propatomic': sel('atom', ['neg', 'propatomic'], ['ops', 'prop', 'cls']),
                  'start': [opt('sign'), 'prop']
              }
              )
