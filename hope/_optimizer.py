# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

# TODO: remove comment!
# replace pow(a, number(int)) by multiplications
# find subexpressions and factor them out => make this blockwise (for that merge 1d expressions into one block)

# 1: merge subexpressions
# 2: create blacklist out of find altered values 
# 3: factor out subexpressions

from __future__ import print_function, division, absolute_import, unicode_literals


from hope._ast import *
from hope._const import *
from hope._dump import Dumper

import sympy as sp

# http://docs.sympy.org/latest/modules/functions/elementary.html
SYM_UNARY_OPERATORS = {}
SYM_UNARY_OPERATORS["+"] = lambda a: +a
SYM_UNARY_OPERATORS["-"] = lambda a: -a

SYM_BINARY_OPERATORS = {}
SYM_BINARY_OPERATORS["+"] = lambda a, b: a + b
SYM_BINARY_OPERATORS["-"] = lambda a, b: a - b
SYM_BINARY_OPERATORS["*"] = lambda a, b: a * b
SYM_BINARY_OPERATORS["/"] = lambda a, b: a / b
SYM_BINARY_OPERATORS["**"] = lambda a, b: a ** b
# SYM_BINARY_OPERATORS["%"] = lambda a, b: a % b
# SYM_BINARY_OPERATORS["<<"] = lambda a, b: a << b
# SYM_BINARY_OPERATORS[">>"] = lambda a, b: a >> b
# SYM_BINARY_OPERATORS["|"] = lambda a, b: a | b
# SYM_BINARY_OPERATORS["^"] = lambda a, b: a ^ b
# SYM_BINARY_OPERATORS["&"] = lambda a, b: a & b

SYM_COMPARE_OPERATORS = {}
SYM_COMPARE_OPERATORS["=="] = lambda a, b: a == b
SYM_COMPARE_OPERATORS["!="] = lambda a, b: a != b
SYM_COMPARE_OPERATORS["<"] = lambda a, b: a < b
SYM_COMPARE_OPERATORS["<="] = lambda a, b: a <= b
SYM_COMPARE_OPERATORS[">"] = lambda a, b: a > b
SYM_COMPARE_OPERATORS[">="] = lambda a, b: a >= b

SYM_NUMBER = {}
SYM_NUMBER[bool] = lambda a: sp.BooleanTrue if a else sp.BooleanFalse
SYM_NUMBER[int] = lambda a: sp.Integer(a)
SYM_NUMBER[float] = lambda a: sp.Float(a, -np.log10(np.finfo(np.float_).resolution))

SYM_UNARY_FUNCTIONS = {}
SYM_UNARY_FUNCTIONS["sin"] = sp.sin
SYM_UNARY_FUNCTIONS["cos"] = sp.cos
SYM_UNARY_FUNCTIONS["tan"] = sp.tan

class CreateExprVisitor(NodeVisitor):
    def __init__(self, dumper): 
        self.dumper = dumper
    def visit_Number(self, node):
        return SYM_NUMBER[node.dtype](node.value)
    def visit_Variable(self, node):
        if not self.dumper.visit(node) in self.symbols:
            raise Exception("Unknown expression: {0!s}".format(self.dumper.visit(node)))
        return sp.Symbol(self.dumper.visit(node))
    def visit_Object(self, node):
        raise Exception("Unable to create expression from : {0!s}".format(self.dumper.visit(node)))
    def visit_ObjectAttr(self, node):
        if not self.dumper.visit(node) in self.symbols:
            raise Exception("Unknown expression: {0!s}".format(self.dumper.visit(node)))
        return sp.Symbol(self.dumper.visit(node))
    def visit_View(self, node):
        if not self.dumper.visit(node.variable) in self.symbols:
            raise Exception("Unknown expression: {0!s}".format(self.dumper.visit(node)))
        self.symbols[self.dumper.visit(node)] = node
        return sp.Symbol(self.dumper.visit(node))
    def visit_UnaryOp(self, node):
        return SYM_UNARY_OPERATORS[node.op](self.visit(node.operand))
    def visit_BinOp(self, node):
        return SYM_BINARY_OPERATORS[node.op](self.visit(node.left), self.visit(node.right))
    def visit_Compare(self, node):
        return SYM_COMPARE_OPERATORS[node.op](self.visit(node.left), self.visit(node.comparator))
    def visit_Call(self, node):
        return self.visit(node.name)(*[self.visit(arg) for arg in node.args])
    # TODO: implement this!
    # def visit_GlobalFunction(self, node): ???
    # TODO: implement this!
    # def visit_HopeAttr(self, node): ???
    def visit_NumpyAttr(self, node):
        if node.name in SYM_UNARY_FUNCTIONS:
            return SYM_UNARY_FUNCTIONS[node.name]
        else:
            return sp.Function('np.' + node.name)
    # TODO: implement this!
    # def visit_NumpyContraction(self, node): ???

class CheckOptimizeVisitor(NodeVisitor):

    def visit_Number(self, node): return True
    def visit_Variable(self, node): return True
    def visit_Object(self, node): return True
    def visit_ObjectAttr(self, node): return True
    # TODO: can wie optimize the index in the extent?
    def visit_View(self, node): return True
    def visit_Expr(self, node): return False
    def visit_Assign(self, node): return False
    def visit_AugAssign(self, node): return False
    def visit_UnaryOp(self, node): return self.visit(node.operand)
    def visit_BinOp(self, node): 
        return self.visit(node.left) and self.visit(node.right) if node.op in SYM_BINARY_OPERATORS else False
    def visit_BoolOp(self, node): return False
    def visit_Compare(self, node): return self.visit(node.left) and self.visit(node.comparator)

    def visit_If(self, node): return False
    def visit_For(self, node): return False
    def visit_While(self, node): return False

    def visit_Call(self, node):
        return self.visit(node.name) and np.all([self.visit(value) for value in list(node.keywords.values())]
                                                + [self.visit(arg) for arg in node.args])

    def visit_GlobalFunction(self, node): return True
    def visit_HopeAttr(self, node): return False

    def visit_NumpyAttr(self, node): 
        return node.name in SYM_UNARY_FUNCTIONS

    def visit_NumpyContraction(self, node): return False
    def visit_Allocate(self, node): return False
    def visit_Return(self, node): return False
    def visit_Block(self, node): return False
    def visit_Body(self, node): return False

class SympyPowVisitor(object):
    def visit(self, expr):
        if expr.is_Pow and expr.exp.is_Integer:
            return [expr]
        else:
            return [item for arg in expr.args for item in self.visit(arg)]

class SympyToAstVisitor(object):

    def __init__(self):
        for name in list(SYM_UNARY_FUNCTIONS.keys()):
            if not name in NPY_UNARY_FUNCTIONS and not name in NPY_CAST_FUNCTIONS:
                raise Exception("Unknown Function {0}".format(name))
            setattr(self, "visit_{0}".format(name), self.npUnaryFunction_visit)

    def visit(self, expr):
        if hasattr(self, 'visit_{0}'.format(type(expr).__name__)):
            return getattr(self, 'visit_{0}'.format(type(expr).__name__))(expr)
        else:
            return self.generic_visit(expr)

    def generic_visit(self, expr):
        raise Exception("Not Implemented Expression: {0}: {1!s}".format(type(expr).__name__, expr))

    def visit_Add(self, expr):
        ret = self.visit(expr.args[0])
        for term in expr.args[1:]:
            node = self.visit(term)
            if isinstance(node, UnaryOp) and node.op == UNARY_OPERATORS["USub"]:
                ret = BinOp("Sub", ret, node.operand)
            else:
                ret = BinOp("Add", ret, node)
        return ret

    def visit_Mul(self, expr):
        sign, numerators, denominator = 1, [], []
        for term in expr.as_ordered_factors():
            if term.is_Pow and term.exp.is_Rational and term.exp.is_negative:
                denominator.append(sp.Pow(term.base, -term.exp, evaluate=term.exp==-1))
            elif term.is_Rational:
                if term.p == -1:
                    sign *= -1
                elif term.p != 1:
                    numerators.append(sp.Rational(term.p))
                if term.q == -1:
                    sign *= -1
                elif term.q != 1:
                    denominator.append(sp.Rational(term.q))
            else:
                numerators.append(term)
        if len(numerators) == 0:
            ret = Number(1)
        else:
            ret = self.visit(numerators[0])
            for arg in numerators[1:]:
                ret = BinOp("Mult", ret, self.visit(arg))
        if len(denominator) > 0:
            ret = BinOp("Div", ret, self.binOp_visit("Mult", denominator))
        if sign < 0:
            ret = UnaryOp("USub", ret)
        return ret

    def visit_Pow(self, expr):
        from sympy.core.singleton import S
        if expr.exp.is_real and float(int(expr.exp)) == expr.exp:
            print("Integer exponent as flaot: {0!s}".format(expr))
        if expr.exp is S.Half or (expr.exp.is_real and expr.exp == 0.5):
            return Call(NumpyAttr("sqrt"), True, [self.visit(expr.base)])
        elif -expr.exp is S.Half or (expr.exp.is_real and -expr.exp == 0.5):
            return BinOp("Div", Number(1.), Call(NumpyAttr("sqrt"), True, [self.visit(expr.base)], {}))
        elif expr.exp == -1:
            return BinOp("Div", Number(1.), self.visit(expr.base))
        else:
            return self.binOp_visit("Pow", expr.args)

    def visit_Integer(self, expr):
        return Number(expr.p)
    def visit_Rational(self, expr):
        return BinOp("Div", Number(float(expr.p)), Number(float(expr.q)))
    def visit_Float(self, expr):
        return Number(float(expr))
    def visit_Symbol(self, expr):
        if not expr.name in self.symbols:
            raise Exception("Unknown symbol: {0!s}".format(expr.name))
        return self.symbols[expr.name]
    def visit_NegativeOne(self, expr):
        return Number(-1)
    def visit_Zero(self, expr):
        return Number(0)
    def visit_One(self, expr):
        return Number(1)

    def npUnaryFunction_visit(self, expr):
        return Call(NumpyAttr(type(expr).__name__), True, [self.visit(arg) for arg in expr.args], {})
    def binOp_visit(self, name, args):
        ret = self.visit(args[0])
        for arg in args[1:]:
            ret = BinOp(name, ret, self.visit(arg))
        return ret

class Optimizer(NodeVisitor):
    def __init__(self):
        self.dumper = Dumper()
        self.checkVisitor, self.createExpr, self.sympyToAst, self.sympyPow, self.next = CheckOptimizeVisitor(), CreateExprVisitor(self.dumper), SympyToAstVisitor(), SympyPowVisitor(), 0

    def visit_Number(self, node): pass
    def visit_NewVariable(self, node): pass
    def visit_Variable(self, node): pass
    def visit_Object(self, node): pass
    def visit_ObjectAttr(self, node): pass
    def visit_Dimension(self, node): pass
    def visit_View(self, node): pass
    def visit_Expr(self, node): pass
    def visit_Assign(self, node): pass
    def visit_AugAssign(self, node): pass
    def visit_UnaryOp(self, node): pass
    def visit_BinOp(self, node): pass
    def visit_Compare(self, node): pass

    def visit_If(self, node):
        # TODO: optimize condition
        # if condition is compile time -> remove!
        self.visit(node.body)
        if not node.orelse is None:
            self.visit(node.orelse)

    def visit_For(self, node):
        self.symbols[self.dumper.visit(node.iter)] = node.iter
        self.visit(node.body)
        del self.symbols[self.dumper.visit(node.iter)]
    def visit_While(self, node):
        self.visit(node.body)

    def visit_Call(self, node): pass
    def visit_GlobalFunction(self, node): pass
    def visit_HopeAttr(self, node): pass
    def visit_NumpyAttr(self, node): pass
    def visit_NumpyContraction(self, node): pass

    def visit_Allocate(self, node):
        self.symbols[self.dumper.visit(node.variable)] = node.variable

    def visit_Return(self, node): pass

    def visit_Block(self, node):
        body, knownexprs, powexprs = [], {}, {}
        for astexpr in node.body:
            self.visit(astexpr)
            if isinstance(astexpr, Assign):
                if isinstance(astexpr.target, View):
                    self.symbols[self.dumper.visit(astexpr.target.variable)] = astexpr.target
                elif isinstance(astexpr.target, Variable):
                    self.symbols[self.dumper.visit(astexpr.target)] = astexpr.target
                else:
                    raise Exception("Unknown token".format(self.dumper.visit(astexpr.target)))
            # TODO: implement for expr
            # TODO: replace subexpressions over several lines
            if isinstance(astexpr, (Assign, AugAssign)) and self.checkVisitor.visit(astexpr.value):
                symexpr = sp.simplify(self.createExpr.visit(astexpr.value))
                subexprs, newexprs = sp.cse(symexpr, optimizations='basic')
                if len(newexprs) != 1:
                    raise Exception("Error running Common Subexpression Detection for {1!s}".format(symexpr))
                newexpr = newexprs[0]
                for symbol, subexpr in subexprs:
                    for subsymbol, newsymbol in list(knownexprs.items()):
                        subexpr = subexpr.subs(subsymbol, newsymbol)
                    for powexpr in self.sympyPow.visit(subexpr):
                        subexpr, _ = self.replace_pow(body, subexpr, powexprs, powexpr, np.abs(powexpr.exp.p))
                    value = self.sympyToAst.visit(sp.simplify(subexpr))
                    name, self.next = "__sp{0}".format(self.next), self.next + 1
                    self.symbols[name] = Variable(name, copy.deepcopy(value.shape), value.dtype)
                    body.append(Assign(self.symbols[name], value))
                    knownexprs[symbol] = sp.Symbol(name)
                    newexpr = newexpr.subs(symbol, knownexprs[symbol])
                for powexpr in sorted(self.sympyPow.visit(newexpr), key=lambda x: -np.abs(x.exp.p)):
                    newexpr, _ = self.replace_pow(body, newexpr, powexprs, powexpr, np.abs(powexpr.exp.p))
                newvalue = self.sympyToAst.visit(sp.simplify(newexpr))
                if astexpr.value.dtype != newvalue.dtype:
                    if isinstance(newvalue, Number):
                        newvalue = Number(astexpr.value.dtype(newvalue.value))
                    else:
                        raise Exception("dtype does not match {0} != {1}".format(self.dumper.visit(astexpr.value), self.dumper.visit(newvalue)))
                if not(len(astexpr.target.shape) > 0 and len(newvalue.shape) == 0):
                    if len(astexpr.value.shape) != len(newvalue.shape):
                        raise Exception("length of shape does not match {0} != {1}".format(self.dumper.visit(astexpr.value), self.dumper.visit(newvalue)))
                    for extent1, extent2 in zip(astexpr.value.shape, newvalue.shape):
                        (lower1, upper1), (lower2, upper2) = extent1, extent2
                        if not ((lower1 is None and lower2 is None) or lower1 == lower2) or upper1 != upper2:
                            raise Exception("shape does not match {0} != {1}".format(self.dumper.visit(astexpr.value), self.dumper.visit(newvalue)))
                astexpr.value = newvalue
                body.append(astexpr)
            else:
                body.append(astexpr)
        node.body = body

    def visit_Body(self, node):
        for block in node.blocks:
            self.visit(block)

    def visit_FunctionDef(self, node):
        if not node.optimized:
            self.symbols = {}
            for var in node.signature:
                self.add_symbol(var)
            self.createExpr.symbols, self.sympyToAst.symbols = self.symbols, self.symbols
            node.optimized = True
            self.visit(node.body)

    def visit_Module(self, node):
        for fktcls in list(node.functions.values()):
            for fkt in fktcls:
                self.visit(fkt)

    def add_symbol(self, symbol):
        if isinstance(symbol, Object):
            for attr in list(symbol.attrs.values()):
                self.add_symbol(attr)
        else:
            self.symbols[self.dumper.visit(symbol)] = symbol

    def replace_pow(self, body, symexpr, powexprs, expr, exp):
        if exp == 1:
            return (symexpr, None)
        elif not (expr.base, exp) in powexprs:
            if exp == 2:
                operand = sp.simplify(expr.base)
                value = BinOp("Mult", self.sympyToAst.visit(operand), self.sympyToAst.visit(operand))
            elif exp % 2 == 1:
                _, operand = self.replace_pow(body, symexpr, powexprs, expr, exp - 1)
                value = BinOp("Mult", self.symbols[operand], self.sympyToAst.visit(sp.simplify(expr.base)))
            else:
                _, operand = self.replace_pow(body, symexpr, powexprs, expr, exp / 2)
                value = BinOp("Mult", self.symbols[operand], self.symbols[operand])
            name, self.next = "__sp{0}".format(self.next), self.next + 1
            self.symbols[name] = Variable(name, copy.deepcopy(value.shape), value.dtype)
            body.append(Assign(self.symbols[name], value))
            powexprs[(expr.base, exp)] = name
        if np.abs(expr.exp.p) == exp:
            symbol = sp.Symbol(powexprs[(expr.base, exp)])
            symexpr = symexpr.subs(expr, self.symbols[powexprs[(expr.base, exp)]].dtype(1) / symbol if expr.exp.is_negative else symbol)
        return (symexpr, powexprs[(expr.base, exp)])
