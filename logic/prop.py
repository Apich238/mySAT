from logic.myPEG import *


class Form:

    def __str__(self):
        return self.repr()

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return hash(self)==hash(other)

    def __hash__(self):
        return self.prefixstr().__hash__()#self.__repr__().__hash__()

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
            return ConForm(Form.compile_ast(node.childs[0]), Form.compile_ast(node.childs[1]))
        elif node.symbol == 'dis':
            return DisForm(Form.compile_ast(node.childs[0]), Form.compile_ast(node.childs[1]))
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
            ptrn = re.compile("[vpx][0-9]+")
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
        return 'N{}'.format( self.value.prefixstr())


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
                  'propcon': ['propatomic', zom(['conj', 'propatomic'])],
                  'propatomic': sel('atom', ['neg', 'propatomic'], ['ops', 'prop', 'cls']),
                  'start': [opt('sign'), 'prop']
              }
              )
#
# import re
# import collections
# from logic.myPEG import *
#
# class Form:
#     def __init__(self, members=[]):
#         self.members = members
#
#     def __str__(self):
#         return self.__repr__()
#
#     def lit(self):
#         return None
#
#     def reduce(self):
#         return self
#
#     @staticmethod
#     def Parse(ln: str):
#         # parsing
#         Token = collections.namedtuple('Token', ['type', 'value', 'pos'])
#
#         def tokenize(line):
#             keywords = {'T', 'F'}
#             token_specification = [
#                 ('NEG', r'\!'),
#                 ('CONJ', r'\&\&'),
#                 ('DISJ', r'\|\|'),
#                 ('IMPL', r'\=\>'),
#                 ('BIIMPL', r'\<\>'),
#                 ('OPSCOBE', r'\('),
#                 ('CLSCOBE', r'\)'),
#                 ('NAME', r'[A-Za-z][A-Za-z0-9]*'),
#                 ('MISMATCH', r'.'),
#             ]
#             tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
#             line_num = 1
#             line_start = 0
#             for mo in re.finditer(tok_regex, line):
#                 kind = mo.lastgroup
#                 value = mo.group()
#                 column = mo.start() - line_start
#
#                 if kind == 'MISMATCH':
#                     raise RuntimeError('{value!r} unexpected on line {line_num}')
#
#                 yield Token(kind, value, column)
#
#         tokens = list(tokenize(ln))
#
#         postfix = []
#         opstack = []
#         precs = {
#             'OPSCOBE': 5,
#             'NEG': 4,
#             'CONJ': 3,
#             'DISJ': 2,
#             'IMPL': 1,
#             'BIIMPL': 0}
#         for i, token in enumerate(tokens):
#             if token[0] in ['NAME', 'T', 'F']:
#                 token = Token('NAME', token[1], token[2])
#                 postfix.append(token)
#             elif token[0] in ['NEG', 'CONJ', 'DISJ', 'IMPL', 'BIIMPL']:
#                 while len(opstack) > 0 and opstack[-1][0] != 'OPSCOBE' and (
#                         precs[opstack[-1][0]] > precs[token[0]] or
#                         precs[opstack[-1][0]] == precs[token[0]] and
#                         opstack[-1][0] in ['CONJ', 'DISJ', 'IMPL', 'BIIMPL']):
#                     postfix.append(opstack.pop())
#                 opstack.append(token)
#             elif token[0] == 'OPSCOBE':
#                 opstack.append(token)
#             elif token[0] == 'CLSCOBE':
#                 k = 0
#                 while len(opstack) > 0 and opstack[-1][0] != 'OPSCOBE':
#                     postfix.append(opstack.pop())
#                     k = k + 1
#                 if opstack[-1][0] == 'OPSCOBE':
#                     opstack.pop()
#                 else:
#                     raise RuntimeError('Scobe error')
#
#         while len(opstack) > 0:
#             postfix.append(opstack.pop())
#
#         stack = []
#         while len(postfix) > 0:
#             token = postfix.pop(0)
#             if token[1] in ['T', 'True']:
#                 stack.append(ConForm())
#             elif token[1] in ['F', 'False']:
#                 stack.append(DisForm())
#             elif token[0] == 'NAME':
#                 stack.append(AtomForm(token[1]))
#             elif token[0] == 'NEG':
#                 stack.append(NegForm(stack.pop()))
#             elif token[0] == 'CONJ':
#                 l = [stack.pop(), stack.pop()]
#                 l.reverse()
#                 stack.append(ConForm(l))
#             elif token[0] == 'DISJ':
#                 l = [stack.pop(), stack.pop()]
#                 l.reverse()
#                 stack.append(DisForm(l))
#             elif token[0] == 'IMPL':
#                 l = [stack.pop(), stack.pop()]
#                 l.reverse()
#                 stack.append(DisForm([NegForm(l[0]), l[1]]))
#             elif token[0] == 'BIIMPL':
#                 l = [stack.pop(), stack.pop()]
#                 l.reverse()
#                 stack.append(DisForm([ConForm([l[0], l[1]]), ConForm([NegForm(l[0]), NegForm(l[1])])]))
#         return stack.pop()
#
#     def __hash__(self):
#         return self.__repr__().__hash__()
#
#     def __eq__(self, other):
#         return hash(self) == hash(other) or str(self) == str(other) or str(self.reduce()) == str(other.reduce())
#
#
# class AtomForm(Form):
#     def __init__(self, name):
#         self.name = name
#         self.members = None
#
#     framed = False
#
#     def copy(self):
#         return AtomForm(self.name)
#
#     def __repr__(self, br=False):
#         if self.framed:
#             m = '[{}]'
#         else:
#             m = '{}'
#         return m.format(str(self.name))
#
#
# class NegForm(Form):
#     def __init__(self, value):
#         self.value = value
#
#     def reduce(self):
#         t = type(self.value)
#         if t is AtomForm:
#             return self
#         elif t is NegForm:
#             return self.value.value.reduce()
#         else:
#             m = [NegForm(memb).reduce() for memb in self.value.members]
#             if t is ConForm:
#                 r = DisForm(m)
#             else:
#                 r = ConForm(m)
#             # print(self,'2',r.reduce())
#             return r.reduce()
#
#     def lit(self):
#         return self.value.lit()
#
#     def __repr__(self, br=False):
#         return ('!{}' if type(self.value) in [AtomForm, NegForm] else '!({})').format(str(self.value))
#
#
# class SetForm(Form):
#     def lit(self):
#         return set.union(*[a.lit() for a in self.members])
#
#
# class ConForm(SetForm):
#     def __init__(self, members=[]):
#         m = []
#         for mm in members:
#             if type(mm) is ConForm and len(mm.members) > 0:
#                 m.extend(mm.members)
#             else:
#                 m.append(mm)
#         self.members = m.copy()
#
#     def reduce(self):
#         r = [memb.reduce() for memb in self.members]
#         r1 = []
#         for t in r:
#             if str(t) == 'False':
#                 return DisForm([])
#             if str(t) != 'True':
#                 r1.append(t)
#         return ConForm(r1)
#
#     def __repr__(self, br=False):
#         return 'True' if len(self.members) == 0 else ('({})' if br else '{}').format(
#             '&&'.join([a.__repr__(br=(type(a) == DisForm)) for a in self.members]))
#
#
# class DisForm(SetForm):
#     def __init__(self, members=[]):
#         m = []
#         for mm in members:
#             if type(mm) is DisForm and len(mm.members) > 0:
#                 m.extend(mm.members)
#             else:
#                 m.append(mm)
#         self.members = m.copy()
#
#     def reduce(self):
#         # return DisForm([memb.reduce() for memb in self.members])
#
#         r = [memb.reduce() for memb in self.members]
#         r1 = []
#         for t in r:
#             if str(t) == 'True':
#                 return ConForm([])
#             if str(t) != 'False':
#                 r1.append(t)
#         return DisForm(r1)
#
#     def __repr__(self, br=False):
#         return 'False' if len(self.members) == 0 else ('({})' if br else '{}').format(
#             '||'.join([str(a) for a in self.members]))
#
#
#
#
#
#
# # res = [Form.Parse("A||B"), Form.Parse("A&&B"), Form.Parse("A>>B"), Form.Parse("!A<>B"), Form.Parse("C&&(A||B)"),
# #        Form.Parse("!C&&!(A<>!B||T)")]
# #
# # print(res)
# # print([r.reduce() for r in res])
# # print([r.reduce().reduce() for r in res])
#
#
# # formula ::= atom |
# #   (formula) |
# #   formula || formula |
# #   formula && formula |
# #   formula >> formula |
# #   -- formula |
# #   QA var formula |
# #   QE var formula
# # atom ::= name( arglist )
# # arglist ::= arg | arg, arglist
# # arg ::= const | function | var
# # const ::= "name"
# # function ::= name( arglist )
# # var :: name
