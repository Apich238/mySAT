import re
import collections


# region peg ops


class oom:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return '{}+'.format(self.x)


class zom:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return '{}*'.format(self.x)


class opt:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return '{}?'.format(self.x)


class chk:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return '&{}'.format(self.x)


class non:
    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return '!{}'.format(self.x)


class sel:
    def __init__(self, *alternatives):
        self.alternatives = list(alternatives)

    def __repr__(self):
        return ' / '.join([str(x) for x in self.alternatives])


# endregion

class TNode:
    def __init__(self, symbol):
        self.symbol = symbol
        self.childs = []

    def add(self, child):
        if not child is None:
            self.childs.append(child)

    def __repr__(self):
        if self.symbol is None or isinstance(self.symbol, list):
            return '{}'.format(self.childs)
        elif len(self.childs) == 0:
            return '{}'.format(self.TokRepr(self.symbol))
        elif len(self.childs) == 1:
            return '{}/{}'.format(self.TokRepr(self.symbol), self.childs[0])
        else:
            return '{}:{}'.format(self.TokRepr(self.symbol), self.childs)

    def TokRepr(self, val):
        if isinstance(val, Token):
            return '{} {}'.format(val.type, val.value)
        else:
            return str(val)

    def TreeRepr(self, ts=0):
        print('\t' * ts + self.TokRepr(self.symbol))
        for c in self.childs:
            c.TreeRepr(ts + 1)


Token = collections.namedtuple('Token', ['type', 'value', 'pos'])


def packrat_memoization(func):
    def mem_wrapper(self, rule, toks, i):

        key = (i, str(rule).__hash__())
        if key in self.mem:
            return self.mem[key]
        else:
            res = func(self, rule, toks, i)
            self.mem[key] = res
            return res

        # key = (i, str(rule).__hash__())
        # if key in self.mem:
        #     if self.mem[key] == 'evaluating':
        #         return None
        #     return self.mem[key]
        # else:
        #     self.mem[key] = 'evaluating'
        #     res = func(self, rule, toks, i)
        #     self.mem[key] = res
        #     return res

    return mem_wrapper


class PEG:
    def __init__(self, start: str, terms: dict, rules: dict):
        self.token_rules = terms.copy()
        self.terms = set(self.token_rules.keys())
        self.rules = rules.copy()
        self.nonterms = set(self.rules.keys())
        self.start = start
        self.mem = {}

    def _tokenize(self, line):
        '''
        Токенизация
        :param line: строка
        :return:
        '''
        # tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in self.terms_spec)
        tok_regex = '|'.join('(?P<{}>{})'.format(k, self.token_rules[k]) for k in self.token_rules)
        tok_regex += '|(?P<MISMATCH>.)'
        line_start = 0
        for mo in re.finditer(tok_regex, line):
            kind = mo.lastgroup
            value = mo.group()
            column = mo.start() - line_start

            if kind == 'MISMATCH':
                raise RuntimeError('value {} unexpected at {}'.format(value, line_start))

            yield Token(kind, value, column)

    @packrat_memoization
    def _parse(self, symbol, tokens, i):
        '''
        функция PEG разбора заданной грамматики
        :param symbol: Текущий разбираемый терминал, нетерминал или правило
        :param tokens: строка токенов
        :param i: текущая позичия указателя
        :return: дерево синтаксического разбора
        '''
        if i > len(tokens):
            return None
        if isinstance(symbol,str) and symbol in self.terms:
            if i < len(tokens) and tokens[i].type == symbol:
                res = TNode(tokens[i])
                return res, i + 1
            else:
                return None
        elif isinstance(symbol,str) and symbol in self.nonterms:
            rule = self.rules[symbol]
        elif isinstance(symbol,(str, list, chk, non, opt, zom, oom, sel)):
            rule = symbol
        else:
            raise ValueError()
        if isinstance(rule,str):
            res = TNode(symbol)
            se = self._parse(rule, tokens, i)
            if se is None:
                return None
            res.add(se[0])
            return res, se[1]
        elif isinstance(rule,list):
            j = i
            res = TNode(symbol if isinstance(symbol, str) else 'seq')
            for sr in rule:
                tres = self._parse(sr, tokens, j)
                if tres is None:
                    return None
                else:
                    subtree, j = tres
                    res.add(subtree)
            return res, j
        elif isinstance(rule,sel):
            alternatives = rule.alternatives
            for a in alternatives:
                altres = self._parse(a, tokens, i)
                if not altres is None:
                    return altres
            return None
        elif isinstance(rule,non):
            tmp = self._parse(rule.x, tokens, i)
            if tmp is None:
                return TNode(None), i
            else:
                return None
        elif isinstance(rule,chk):
            tmp = self._parse(rule.x, tokens, i)
            if tmp is None:
                return None
            else:
                return TNode(None), i
        elif isinstance(rule,zom):
            res = TNode('zom')
            j = i
            while True:
                tr = self._parse(rule.x, tokens, j)
                if tr is None:
                    break
                res.add(tr[0])
                j = tr[1]
            if len(res.childs) == 0:
                res = None
            return res, j
        elif isinstance(rule,opt):
            optional = self._parse(rule.x, tokens, i)
            if optional is None:
                return None, i
            else:
                return optional
        elif isinstance(rule,oom):
            one = self._parse(rule.x, tokens, i)
            if one is None:
                return None
            res = TNode('oom')
            res.add(one[0])
            j = one[1]
            while True:
                tr = self._parse(rule.x, tokens, j)
                if tr is None:
                    break
                res.add(tr[0])
                j = tr[1]
            if len(res.childs) == 0:
                res = None
            return res, j
        else:
            raise ValueError()

    def Parse(self, line):
        '''
        Разбор строки
        :param line:
        :return:
        '''
        tokens = list(self._tokenize(line))
        tokens = [t for t in tokens if t.type != 'ignore']
        res = self._parse(self.start, tokens, 0)
        self.mem = {}
        if res is None:
            return None
        elif len(tokens) > res[1]:
            raise ValueError('остались лишние токены: {}'.format(tokens[res[1]:]))
        else:
            return res[0]

