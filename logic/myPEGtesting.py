from myPEG import *

if __name__ == '__main__':
    # p = PEG('S',
    #         {
    #             'a': 'a'
    #         },
    #         {
    #             'S': 'A',
    #             'A': sel(['A', 'a'], 'a')
    #         })
    #
    # print(p.Parse('aaaaa'))
    # Грамматика целочисленной арифметики
    arith = PEG('expr',
                {
                    'num': r'[0-9]+(\.[0-9]*)?',
                    'opbracket': r'\(',
                    'clbracket': r'\)',
                    'plus': r'\+',
                    'minus': r'\-',
                    'prod': r'\*',
                    'div': r'\/',
                    'ignore': r' +'
                },
                {
                    'expr': 'sum',
                    'sum': ['product', zom([sel('plus', 'minus'), 'product'])],
                    'product': ['value', zom([sel('prod', 'div'), 'value'])],
                    'value': [opt('minus'), sel('num', ['opbracket', 'sum', 'clbracket'])]
                })


    def parsingTree2ASTree(node: TNode):
        if len(node.childs) == 1:
            return parsingTree2ASTree(node.childs[0])
        elif node.symbol == 'value' and len(node.childs) == 2:
            tr = TNode(node.childs[0].symbol)
            tr.add(parsingTree2ASTree(node.childs[1]))
            return tr
        elif node.symbol in ['product', 'sum']:
            r = TNode(node.symbol)
            r.add(parsingTree2ASTree(node.childs[0]))
            zcs = node.childs[1].childs
            for seq in zcs:
                for c in seq.childs:
                    if type(c) is Token:
                        r.add(c)
                    else:
                        r.add(parsingTree2ASTree(c))
            return r
        elif node.symbol == 'seq' and node.childs[0].symbol.type == 'opbracket':
            return parsingTree2ASTree(node.childs[1])
        elif True:
            return node


    def eval_arythm(node):
        if type(node.symbol) is Token and node.symbol.type == 'num':
            return float(node.symbol.value)
        elif node.symbol == 'sum':
            r = eval_arythm(node.childs[0])
            for i in range(2, len(node.childs), 2):
                if node.childs[i - 1].symbol.type == 'minus':
                    r -= eval_arythm(node.childs[i])
                else:
                    r += eval_arythm(node.childs[i])
            return r
        elif node.symbol == 'product':
            r = eval_arythm(node.childs[0])
            for i in range(2, len(node.childs), 2):
                if node.childs[i - 1].symbol.type == 'div':
                    r /= eval_arythm(node.childs[i])
                else:
                    r *= eval_arythm(node.childs[i])
            return r
        elif node.symbol.type == 'minus':
            return -eval_arythm(node.childs[0])
        else:
            print('{}'.format(node))


    def testline(l, peg: PEG):
        print('---------------')
        print(l)
        res = peg.Parse(l)
        print('parsing tree:')
        # res.TreeRepr()
        ast = parsingTree2ASTree(res)
        print('abstract syntax tree:')
        ast.TreeRepr()
        val = eval_arythm(ast)
        print('result:')
        print(val)
        # print()


    ls = [
        '1', '0.', '0.0010', '-1', '1+2', '1*2', '1*-2', '-1*2', '1-2*3',
        '1*2-3', '1-2/3*5+4', '1-2/(3+4)/5',
        '345 - -56 + (1+2)', '1*2*3/(4+5)*6+-7-2',
        '(876-787+(765-234)*2736/23/123)/23*76*5-1'
    ]
    for l in ls:
        testline(l, arith)
# formula ::= atom |
#   (formula) |
#   formula || formula |
#   formula && formula |
#   formula >> formula |
#   -- formula |
#   QA var formula |
#   QE var formula
# atom ::= name( arglist )
# arglist ::= arg | arg, arglist
# arg ::= const | function | var
# const ::= "name"
# function ::= name( arglist )
# var :: name
# predp = PEG('expr',
#             {
#                 'NEG': r'\-',
#                 'CONJ': r'\&',
#                 'DISJ': r'\|',
#                 'IMPL': r'\>',
#                 'ALLQ': r'\@',
#                 'EXQ': r'\#',
#                 'OPRNTH': r'\(',
#                 'CPRNTH': r'\)',
#                 'TRUTH': r'[1T]',
#                 'FALSE': r'[0F]',
#                 'NAME': r'[A-Za-z][A-Za-z0-9]*',
#                 'COMMA': r','
#             },
#             {
#                 'expr': sel('atomic', ['expr', 'IMPL', 'expr'], ['expr', 'DISJ', 'expr'], ['expr', 'CONJ', 'expr']),
#                 'atomic': sel('atom', ['NEG', 'atomic'], ['OPRNTH', 'expr', 'CPRNTH']),
#                 'atom': sel('TRUTH', 'FALSE', 'qexpression', 'predicate'),
#                 'qexpression': ['quantifier', 'name', 'predicate'],
#             })
#
#
# for l in ls:
#     testline(l, predp)
