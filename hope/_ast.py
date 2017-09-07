# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>
"""
Contains all the HOPE AST nodes currently supported

"""

from __future__ import print_function, division, absolute_import, unicode_literals


import ast
import copy

from hope._const import *
from hope.exceptions import UnsupportedFeatureException

class Token(object):
    def getId(self):
        return "{0}{1!s}".format(PY_TYPE_CHAR[self.dtype], "" if len(self.shape) == 0 else len(self.shape))

    def __ne__(self, other):
        return not self == other
    
    
    def __str__(self):
        from hope._dump import Dumper
        return Dumper().visit(self)


class Number(Token):
    def __init__(self, value):
        if not type(value) in PY_C_TYPE:
            raise Exception("Numberformat not implemented: {0}.{1!s}".format(value, type(value)))
        self.value, self.shape, self.dtype = value, [], type(value)
    # TODO: implement cmp

    def __eq__(self, other):
        return isinstance(other, Number) and self.value == other.value and self.dtype == other.dtype


class NewVariable(Token):
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, NewVariable) and self.name == other.name


class Variable(Token):
    def __init__(self, name, shape, dtype = None, scope = "block", allocated = False):
        self.name, self.shape, self.dtype, self.scope, self.allocated = name, shape, dtype, scope, allocated

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name and self.dtype == other.dtype and self.shape == other.shape and self.scope == other.scope


class Object(Token):
    def __init__(self, name, instance, parent=None):
        self.parent, self.name, self.classname, self.instance, self.attrs = parent, name, instance.__class__.__name__, instance, {}

    def getId(self, prefix = "o"):
        chars = "{0}{1}_".format(prefix, self.classname)
        for key in sorted(self.attrs.keys()):
            if isinstance(self.attrs[key], Object):
                chars += self.attrs[key].getId()
            elif len(self.attrs[key].shape) == 0:
                chars += PY_TYPE_CHAR[self.attrs[key].dtype]
            else:
                chars += "{0}{1}".format(PY_TYPE_CHAR[self.attrs[key].dtype], len(self.attrs[key].shape))
        return chars

    def getAttr(self, name):
        if not name in self.attrs:
            value = getattr(self.instance, name)
            if isinstance(value, np.ndarray):
                if value.dtype.isbuiltin != 1:
                    raise Exception("Only Builtin datatypes are supported: in {0} value {1} has type {2!s}".format(self.name, name, value.dtype))
                self.attrs[name] = ObjectAttr(self, name, value.dtype.type, len(value.shape), PY_TYPE_CHAR[value.dtype.type])
            elif hasattr(value, "dtype"):
                self.attrs[name] = ObjectAttr(self, name, value.dtype.type, 0, PY_TYPE_CHAR[value.dtype.type])
            elif isinstance(value, int): 
                self.attrs[name] = ObjectAttr(self, name, int, 0, PY_TYPE_CHAR[int])
            elif isinstance(value, float):
                self.attrs[name] = ObjectAttr(self, name, float, 0, PY_TYPE_CHAR[float])
            else:
                self.attrs[name] = Object(name, value, self)
        return self.attrs[name]

    def __eq__(self, other):
        if (self.parent is None) != (other.parent is None) or (self.parent is not None and other.parent is not None and self.parent != other.parent):
            return False
        return isinstance(other, Object) and self.name == other.name and self.classname == other.classname and self.attrs == other.attrs

    # on deepcopy do not copy the parent object, since we have not information about ist and do not require it to be deepcopy
    # ths violates the python semantichs :(
    def __deepcopy__(self, memo):
        memo[id(self)] = Object.__new__(Object)
        for key, value in list(self.__dict__.items()):
            setattr(memo[id(self)], key, value if key == "instance" and value is not None else copy.deepcopy(value, memo))
        return memo[id(self)]


class ObjectAttr(Token):
    def __init__(self, parent, name, dtype, dims, char):
        self.parent, self.name, self.shape, self.dtype, self.char = parent, name, [(None, Dimension(self, dim)) for dim in range(dims)], dtype, char

    def __eq__(self, other):
        return isinstance(other, ObjectAttr) and self.parent == other.parent and self.name == other.name and self.dtype == other.dtype and self.shape == other.shape and self.char == other.char

    def getTrace(self):
        parent = self.parent
        trace = [self.name]
        while not parent is None:
            trace.insert(0, parent.name)
            parent = parent.parent
        return trace

class Dimension(Token):
    def __init__(self, variable, dim):
        self.variable, self.shape, self.dim = variable, [], dim

    def __eq__(self, other):
        return isinstance(other, Dimension) and self.dim == other.dim \
            and isinstance(self.variable, Variable) and self.variable.name == other.variable.name and self.variable.dtype == other.variable.dtype \
            and len(self.variable.shape) == len(other.variable.shape) and self.variable.scope == other.variable.scope

class DimensionSlice(Dimension):
    def __init__(self, variable, dim, slice):
        super(DimensionSlice, self).__init__(variable, dim)
        self.slice = slice

class View(Token):
    def __init__(self, variable, extents):
        if not isinstance(variable, (Variable, ObjectAttr)):
            raise Exception("A view can only be constructed on a variable or a objectAttr")
        if len(extents) == 0:
            raise Exception("A View needs to have an extent")
        if len(extents) == 1 and extents[0] == (None, None):
            extents = copy.deepcopy(variable.shape)
        if len(variable.shape) != len(extents):
            raise Exception("Extends of variable and subscript do not match")
        self.variable, self.dtype, self.extents, self.shape = variable, variable.dtype, extents, []
        for ind, (variable_extent, extent) in enumerate(zip(self.variable.shape, extents[:])):
            # TODO: check if variable extends and extends do match
            if isinstance(extent, tuple):
                lower, upper = extent
                if lower is None: lower = variable_extent[0]
                if isinstance(lower, Number) and lower.value == 0: lower = None
                if upper is None: upper = variable_extent[1]
                if isinstance(upper, Number) and upper.value < 0: 
                    upper = DimensionSlice(variable, variable_extent[1].dim, upper)
                    extents[ind] = (lower, copy.deepcopy(upper))
                self.shape.append((lower, upper))

    def __eq__(self, other):
        return isinstance(other, View) and self.variable == other.variable and self.dtype == other.dtype and self.shape == other.shape


class Assign(Token):
    def __init__(self, target, value):
        self.target, self.value, self.dtype, self.shape = target, value, target.dtype, target.shape

    def __eq__(self, other):
        return isinstance(other, Assign) and self.target == other.target and self.value == other.value


class AugAssign(Token):
    def __init__(self, op, target, value):
        self.op, self.target, self.value, self.dtype, self.shape = AUGMENTED_ASSIGN[type(op).__name__], target, value, target.dtype, target.shape

    def __eq__(self, other):
        return isinstance(other, AugAssign) and self.op == other.op and self.target == other.target and self.value == other.value


class Expr(Token):
    def __init__(self, value):
        self.value, self.shape = value, value.shape

    def __eq__(self, other):
        return isinstance(other, Expr) and self.value == other.value


class UnaryOp(Token):
    def __init__(self, op, operand):
        opname, dtypefkt = UNARY_OPERATORS[op]
        dtype = None if operand.dtype is None else dtypefkt(operand.dtype)
        self.op, self.operand, self.dtype, self.shape = opname, operand, dtype, operand.shape

    def __eq__(self, other):
        return isinstance(other, UnaryOp) and self.op == other.op and self.operand == other.operand


class BinOp(Token):
    def __init__(self, op, left, right):
        opname, dtypefkt = BINARY_OPERATORS[op]
        dtype = None if left.dtype is None or right.dtype is None else dtypefkt(left.dtype, right.dtype)
        shape = left.shape if len(left.shape) > 0 else right.shape
        self.op, self.left, self.right, self.dtype, self.shape = opname, left, right, dtype, shape

    def __eq__(self, other):
        return isinstance(other, BinOp) and self.op == other.op and self.left == other.left and self.right == other.right


class BoolOp(Token):
    def __init__(self, op, values):
        self.op, self.values, self.dtype, self.shape = BOOL_OPERATORS[op], values, np.bool_, values[0].shape

    def __eq__(self, other):
        return isinstance(other, BoolOp) and self.op == other.op and all([otherval == selfval for otherval, selfval in zip(other.values, self.values)])


class Compare(Token):
    def __init__(self, op, left, comparator):
        opname, dtypefkt = COMPARE_OPERATORS[type(op).__name__]
        dtype = None if left.dtype is None or comparator.dtype is None else dtypefkt(left.dtype, comparator.dtype)
        shape = left.shape if len(left.shape) > 0 else comparator.shape
        self.op, self.left, self.comparator, self.dtype, self.shape = opname, left, comparator, dtype, shape

    def __eq__(self, other):
        return isinstance(other, BinOp) and self.op == other.op and self.left == other.left and self.comparator == other.comparator


class If(Token):
    def __init__(self, test, body, orelse):
        self.test, self.dtype, self.shape, self.body, self.orelse = test, None, [], body, orelse
    def __eq__(self, other):
        return isinstance(other, If) and self.test == other.test and self.body == other.body and self.orelse == other.orelse


class For(Token):
    def __init__(self, iter, lower, upper, body):
        self.iter, self.dtype, self.shape, self.lower, self.upper, self.body = iter, None, [], lower, upper, body

    def __eq__(self, other):
        return isinstance(other, For) and self.iter == other.iter and self.lower == other.lower and self.upper == other.upper and self.body == other.body


class While(Token):
    def __init__(self, test, body):
        self.test, self.dtype, self.shape, self.body = test, np.bool_, [], body

    def __eq__(self, other):
        return isinstance(other, While) and self.test == other.test and self.body == other.body


class Call(Token):
    def __init__(self, name, iterable, args = [], keywords = {}):
        self.name, self.dtype, self.shape, self.args, self.keywords = name, name.get_dtype(args, keywords), name.get_shape(args, keywords) if iterable else [], args, keywords


# TODO: make this nicer, implement return values
class GlobalFunction(Token):
    # TODO: make argid token to change argid by adapting object
    def __init__(self, name, shape, dtype):
        self.name, self.shape, self.dtype = name, shape, dtype

    def get_dtype(self, args, keywords):
        return self.dtype

    def get_shape(self, args, keywords):
        return self.shape


# TODO: make this nicer
class HopeAttr(Token):
    def __init__(self, name):
        self.name, self.shape = name, []

    def __check(self, args, keywords):
        if self.name == "exp":
            if len(args) != 1 or len(keywords) != 0:
                raise Exception("hope.exp requires exaclty one argument")
        else:
            raise Exception("Unknown hope attribute: hope.pow {0!s}".format(self.name))

    def get_dtype(self, args, keywords):
        self.__check(args, keywords)
        if self.name == "exp":
            return args[0].dtype

    def get_shape(self, args, keywords):
        self.__check(args, keywords)
        if self.name == "exp":
            return args[0].shape


# TODO: make this nicer
class NumpyAttr(Token):
    def __init__(self, name):
        self.name, self.shape = name, []

    def __check(self, args, keywords):
        if self.name in ["empty", "zeros", "ones"]:
            if len(args) != 1:
                raise UnsupportedFeatureException("numpy.{0} requires one argument".format(self.name))
            if len(keywords) > 1:
                raise Exception("numpy.{0} requires at most one keyword argument: dtype".format(self.name))
            if len(keywords) == 1 and (
                                       "dtype" not in keywords or \
                                       not isinstance(keywords["dtype"], NumpyAttr) or \
                                       not hasattr(np, keywords["dtype"].name) or \
                                       not getattr(np, keywords["dtype"].name) in PY_C_TYPE):
                raise Exception("numpy.{0} requires a dtype".format(self.name))
            
        elif self.name == "interp":
            for keyword in list(keywords.keys()):
                if not keyword in ["left", "right"]:
                    raise Exception("numpy.interp only allows left and right as keywords")
            if len(args) != 3:
                raise Exception("numpy.interp requires three arguments")
            for idx in [1, 2]:
                if len(args[idx].shape) != 1:
                    raise Exception("numpy.interp requires the second and third argument to be a 1d array")
                
        elif self.name in NPY_UNARY_FUNCTIONS or self.name in NPY_CAST_FUNCTIONS:
            if len(args) != 1 or len(keywords) != 0:
                raise UnsupportedFeatureException("HOPE only supports one argument for 'numpy.%s'. Received: %s"%(self.name, len(args) + len(keywords)))
        else:
            raise UnsupportedFeatureException("Unknown numpy function: numpy.{0}".format(self.name))

    def get_dtype(self, args, keywords):
        self.__check(args, keywords)
        if self.name in ["empty", "zeros", "ones"]:
            return getattr(np, keywords["dtype"].name) if len(keywords) == 1 else np.float64
        elif self.name in ["interp", "sign"] or self.name in NPY_UNARY_FUNCTIONS:
            return args[0].dtype
        elif self.name in NPY_CAST_FUNCTIONS:
            return NPY_CAST_FUNCTIONS[self.name]

    def get_shape(self, args, keywords):
        self.__check(args, keywords)
        if self.name in ["empty", "zeros", "ones"]:
            return [(None, arg) for arg in args[0]] if isinstance(args[0], list) else [(None, args[0])]
        elif self.name in ["interp", "sign"] or self.name in NPY_UNARY_FUNCTIONS or self.name in NPY_CAST_FUNCTIONS:
            return args[0].shape


class NumpyContraction(Token):
    def __init__(self, variable, op, value):
        self.variable, self.op, self.value, self.shape, self.dtype = variable, op, value, [], variable.dtype


class Allocate(Token):
    def __init__(self, variable):
        if variable.dtype is None:
            raise Exception("Allocation requires dtype: {0!s}".format(variable))
        self.variable, self.shape = variable, []


class Return(Token):
    def __init__(self, value):
        self.value, self.shape = value, []


class Block(Token):
    def __init__(self, expr):
        self.body, self.shape, self.merged = [expr], expr.shape, None

class Reference(Token):
    def __init__(self, target, value):
        self.target, self.value, self.dtype, self.shape = target, value, target.dtype, []

    def __eq__(self, other):
        return isinstance(other, Reference) and self.target == other.target and self.value == other.value

class Body(Token):
    def __init__(self, blocks):
        self.blocks, self.dtype, self.shape = blocks, None, []


class FunctionDef(Token):
    def __init__(self, name, signature):
        self.name, self.signature, self.shape, self.dtype, self.merged, self.shapes, self.decl, self.optimized, self.isbound \
            = name, signature, None, None, {}, {}, None, False, len(signature) > 0 and isinstance(signature[0], Object)
        self.return_allocated = True

    def getId(self):
        return "".join([arg.getId() for arg in self.signature])


class Module(Token):
    def __init__(self, main):
        self.main, self.functions = main, {}


class NodeVisitor(object):
    def visit(self, node):
        if hasattr(self, 'visit_{0}'.format(type(node).__name__)):
            return getattr(self, 'visit_{0}'.format(type(node).__name__))(node)
        else:
            return self.generic_visit(node)

    def generic_visit(self, node):
        raise Exception("Not Implemented Token: {0}({1!s})".format(type(node).__name__, node))


def call_has_kw_args(node):
    if hasattr(node, "kwargs"):
        return node.kwargs is not None
    return any(kw.arg is None for kw in getattr(node, "keywords", []))


def call_has_starargs(node):
    if hasattr(node, "starargs"):
        return node.starargs is not None
    try:
        # must be defined in Python 3
        ast.Starred
    except AttributeError:
        raise RuntimeError("should not happen")
    return any(isinstance(arg, ast.Starred) for arg in node.args)


