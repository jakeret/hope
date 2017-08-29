# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

from __future__ import print_function, division, absolute_import, unicode_literals


import ast
import sys
import inspect

from hope._dump import Dumper
from hope._ast import *
import hope._cache as cache
from hope.exceptions import UnsupportedFeatureException


class SymbolNamesCollector(NodeVisitor):
    """
    Traverses a HOPE AST (resp. subtree) to collect variable names (symbols) 
    in order to identify if variables are only used locally (in a loop/block) 
    or multiple times (body scope). 
    """
    
    def visit_Number(self, node):
        return []
    def visit_NewVariable(self, node):
        return [node.name]
    def visit_Variable(self, node):
        symbols = [node.name]
        for upper, lower in node.shape:
            symbols += [] if lower is None else self.visit(lower)
            symbols += [] if upper is None else self.visit(upper)
        return list(set(symbols))
    def visit_Object(self, node): 
        return []
    def visit_ObjectAttr(self, node):
        return []
    def visit_Dimension(self, node):
        if isinstance(node.variable, ObjectAttr):
            return []
        return [node.variable.name]
    def visit_DimensionSlice(self, node):
        return self.visit_Dimension(node)
    def visit_View(self, node):
        symbols = self.visit(node.variable)
        for upper, lower in node.shape:
            symbols += [] if lower is None else self.visit(lower)
            symbols += [] if upper is None else self.visit(upper)
        return list(set(symbols))
    def visit_Expr(self, node):
        return self.visit(node.value)
    def visit_Assign(self, node):
        return list(set(self.visit(node.target) + self.visit(node.value)))
    def visit_Reference(self, node):
        return list(set(self.visit(node.target) + self.visit(node.value)))
    def visit_AugAssign(self, node):
        return list(set(self.visit(node.target) + self.visit(node.value)))
    def visit_UnaryOp(self, node):
        return self.visit(node.operand)
    def visit_BinOp(self, node):
        return list(set(self.visit(node.left) + self.visit(node.right)))
    def visit_BoolOp(self, node):
        return list(set([symbol for value in node.values for symbol in self.visit(value)]))
    def visit_Compare(self, node):
        return list(set(self.visit(node.left) + self.visit(node.comparator)))
    def visit_If(self, node):
        return list(set(self.visit(node.test) + self.visit(node.body) + (self.visit(node.orelse) if not node.orelse is None else [])))
    def visit_For(self, node):
        return list(set(self.visit(node.iter) + self.visit(node.body)))
    def visit_While(self, node):
        return list(set(self.visit(node.test) + self.visit(node.body)))
    def visit_Call(self, node):
        return list(set(self.visit(node.name) + [symbol for arg in node.args + [value for value in list(node.keywords.values())] for symbol in self.visit(arg)]))
    def visit_GlobalFunction(self, node):
        return []
    def visit_HopeAttr(self, node):
        return []
    def visit_NumpyAttr(self, node):
        return []
    def visit_NumpyContraction(self, node):
        return list(set(self.visit(node.variable) + self.visit(node.value)))
    def visit_Allocate(self, node):
        return self.visit(node.variable)
    def visit_Return(self, node):
        return self.visit(node.value)
    def visit_Block(self, node):
        return list(set([symbol for expr in node.body for symbol in self.visit(expr)]))
    def visit_Body(self, node):
        return list(set([symbol for block in node.blocks for symbol in self.visit(block)]))

class AllocateVisitor(NodeVisitor):
    """
    Visits the HOPE AST to identify variables, which needs to be allocated
    """
    
    def __init__(self, variables):
        self.variables = variables
        
    def generic_visit(self, node):
        pass
    
    def visit_If(self, node):
        self.visit(node.body)
        if node.orelse is not None:
            self.visit(node.orelse)
            
    def visit_For(self, node):
        scope = node.iter.scope
        node.iter.scope = "body"
        self.visit(node.body)
        node.iter.scope = scope
        
    def visit_While(self, node):
        self.visit(node.body)
        
    def visit_Block(self, node):
        return [self.visit(block) for block in node.body]
    
    def visit_Body(self, node):
        symbolcollector = SymbolNamesCollector()
        symboldict = {block: set(symbolcollector.visit(block)) for block in node.blocks}

        def check_blocks(start):
            for ind, block in enumerate(node.blocks):
                if start < ind:
                    for other in node.blocks[:ind] + node.blocks[(ind + 1):]:
                        for symbol in symboldict[block].intersection(symboldict[other]):
                            if self.variables[symbol].scope == "block":
                                self.variables[symbol].scope = "body"
                                self.variables[symbol].allocated = True
                                newBlock = Block(Allocate(self.variables[symbol]))
                                symboldict[newBlock] = set(symbolcollector.visit(newBlock))
                                node.blocks = node.blocks[:ind] + [newBlock] + node.blocks[ind:]
                                return ind - 1
            return None

        start = -1
        while True:
            start = check_blocks(start)
            if start is None:
                break

        return [self.visit(block) for block in node.blocks]

class IterableFunctionVisitor(ast.NodeVisitor):
    """
    Analyzes an AST to identify if a called function is 
    applied element-wise on an array (can be vectorized) if 
    the array has to be passed as argument.
    """
    
    def __init__(self, namespace, stack):
        self.namespace, self.stack = namespace, stack
        
    def generic_visit(self, node):
        raise NotImplementedError("Token not implemented: {0}".format(ast.dump(node)))
    def visit_Expr(self, node):
        return self.visit(node.value)
    def visit_Str(self, node):
        return False
    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise Exception("Only single target assignemants are supported")
        return self.visit(node.targets[0]) and self.visit(node.value)
    def visit_AugAssign(self, node):
        return self.visit(node.target) and self.visit(node.value)
    def visit_Subscript(self, node):
        return self.visit(node.value) and self.visit(node.slice)
    def visit_UnaryOp(self, node):
        return self.visit(node.operand)
    def visit_BinOp(self, node):
        return self.visit(node.left) and self.visit(node.right)
    def visit_BoolOp(self, node):
        return all([self.visit(value) for value in node.values])
    def visit_Compare(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise Exception("More than one ops or more than one comparators are not supported")
        return self.visit(node.left) and self.visit(node.comparators[0])
    def visit_Num(self, node):
        return True
    def visit_Name(self, node):
        return True
    def visit_NameConstant(self, node):
        return True
    def visit_Slice(self, node):
        return node.lower if node.lower is None else self.visit(node.lower) and node.upper if node.upper is None else self.visit(node.upper)
    def visit_ExtSlice(self, node):
        return all([self.visit(dim) for dim in node.dims])
    def visit_Index(self, node):
        return True
    def visit_If(self, node):
        return (self.body_visit(node.orelse) if len(node.orelse) > 0 else True) and self.visit(node.test) and self.body_visit(node.body)
    def visit_For(self, node):
        return False
    def visit_While(self, node):
        return False
    def visit_Attribute(self, node):
        import hope
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and isinstance(node.value.ctx, ast.Load) \
                and node.value.id in self.namespace and self.namespace[node.value.id] is hope:
            return True
        elif isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and isinstance(node.value.ctx, ast.Load) \
                and node.value.id in self.namespace and self.namespace[node.value.id] is np:
            return node.attr != "sum"
        else:
            return False

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Name) and isinstance(node.func.ctx, ast.Load)):
            called_fkt = self.namespace[node.func.id]
            if (inspect.isbuiltin(called_fkt) or called_fkt.__name__ == "_hope_callback") and hasattr(cache, str(id(called_fkt))):
                called_fkt = getattr(cache, str(id(called_fkt)))
            elif not inspect.isfunction(called_fkt):
                raise Exception("Function '{0}' not a unbound, pure python function: ({1})".format(self.fkt.__name__, ast.dump(node)))
            
            if not node.func.id in self.namespace or not inspect.isfunction(called_fkt):
                raise Exception("Function '{0}' not accessible form global scope of function: ({1})".format(self.fkt.__name__, ast.dump(node)))
            
            called_fkt_ast = get_fkt_ast(called_fkt)
            
            if called_fkt in self.stack:
                return True
            else:
                return IterableFunctionVisitor(called_fkt.__globals__, self.stack + [called_fkt]).visit(called_fkt_ast)
        elif isinstance(node.func, ast.Attribute):
            return self.visit(node.func)
        else:
            return False

    def visit_Return(self, node):
        return self.visit(node.value)
    def visit_FunctionDef(self, node):
        return self.body_visit(node.body)
    def body_visit(self, stmts):
        for stmt in stmts:
            if not self.visit(stmt):
                return False
        return True

class ASTTransformer(ast.NodeVisitor):
    """
    A visitor implementation, traversing a Python AST of a function to create a HOPE AST.
    
    :param modtoken: The :py:class:`hope._ast.Module`, which should contain the function's HOPE AST.
    """
    
    def __init__(self, modtoken):
        self.modtoken, self.dumper, self.stack, self.exprs, self.next = modtoken, Dumper(), [], None, 0

    def merge_shapes(self, left, right):
        for leftslice, rightslice in zip(left.shape, right.shape):
            if leftslice is None or rightslice is None:
                raise Exception("Invalid shape!")
            
            leftslicestr = "{0}:{1}".format("" if leftslice[0] is None else self.dumper.visit(leftslice[0]), self.dumper.visit(leftslice[1]))
            rightslicestr = "{0}:{1}".format("" if rightslice[0] is None else self.dumper.visit(rightslice[0]), self.dumper.visit(rightslice[1]))
            self.token.shapes[leftslicestr], self.token.shapes[rightslicestr] = leftslice, rightslice
            if leftslicestr != rightslicestr:
                if not leftslicestr in self.merged and not rightslicestr in self.merged:
                    self.merged[leftslicestr] = len(self.merged)
                    self.merged[rightslicestr] = self.merged[leftslicestr]
                elif not leftslicestr in self.merged:
                    self.merged[leftslicestr] = self.merged[rightslicestr]
                elif not rightslicestr in self.merged:
                    self.merged[rightslicestr] = self.merged[leftslicestr]
                elif self.merged[leftslicestr] != self.merged[rightslicestr]:
                    for name, group in list(self.merged.items()):
                        if group == self.merged[rightslicestr]:
                            self.merged[name] = self.merged[leftslicestr]

    def dump_shape(self, shape):
        return "[{0}]".format(",".join(["{0}:{1}".format("" if seg[0] is None else self.dumper.visit(seg[0]), self.dumper.visit(seg[1])) for seg in shape]))

    def generic_visit(self, node):
        raise NotImplementedError("Token not implemented: {0}".format(ast.dump(node)))

    def visit_Expr(self, node):
        try:
            return Expr(self.visit(node.value))
        except Exception as ex:
            from hope._tosource import tosource
            ex.args = ((ex.args[0] if ex.args else "") + "\nin line " + tosource(node),) + ex.args[1:]
            raise

    def visit_Assign(self, node):
        try:
            if len(node.targets) != 1:
                raise UnsupportedFeatureException("Only single target assignments are supported")
            
            target, value = self.visit(node.targets[0]), self.visit(node.value)
            
            if isinstance(target, NewVariable):
                self.variables[target.name] = Variable(target.name, copy.deepcopy(value.shape), value.dtype)
                target = self.variables[target.name]
            elif isinstance(target, Variable) and len(target.shape) == 0: pass
            elif isinstance(target, ObjectAttr): pass
            elif not isinstance(target, View):
                raise Exception("Assignments are only allowed to views or variables")
            
            # TODO: should we check dtypes?
            if len(target.shape) > 0 and len(value.shape) == 0: pass
            elif len(target.shape) != len(value.shape):
                raise Exception("Invalid shapes: {0!s} {1!s}".format(self.dump_shape(target.shape), self.dump_shape(value.shape)))
            
            else:
                self.merge_shapes(target, value)
                

            if isinstance(value, ObjectAttr) and not isinstance(target, View): # reference to member
                name = ".".join(value.getTrace())
                self.variables[name] = Variable(name, shape=copy.deepcopy(value.shape), dtype=value.dtype, scope="body")
                if not isinstance(target, ObjectAttr): # avoid that target is allocated
                    self.variables[target.name].scope = "body"
                    self.variables[target.name].allocated = True
                
                return Reference(target, value)
                
            return Assign(target, value)
        
        except Exception as ex:
            from hope._tosource import tosource
            ex.args = ((ex.args[0] if ex.args else "") + "\nin line " + tosource(node),) + ex.args[1:]
            raise

    def visit_AugAssign(self, node):
        try:
            target, value = self.visit(node.target), self.visit(node.value)
            if len(value.shape) == 0: pass
            elif len(target.shape) != len(value.shape):
                raise Exception("Invalid shapes: {0!s} {1!s}".format(self.dump_shape(target.shape), self.dump_shape(value.shape)))
            
            else:
                self.merge_shapes(target, value)
            # TODO: should we check dtypes?
            return AugAssign(node.op, target, value)
        except Exception as ex:
            from hope._tosource import tosource
            ex.args = ((ex.args[0] if ex.args else "") + "\nin line " + tosource(node),) + ex.args[1:]
            raise

    def visit_Subscript(self, node):
        return View(self.visit(node.value), self.visit(node.slice))

    def visit_UnaryOp(self, node):
        op = type(node.op).__name__
        operand = self.visit(node.operand)
        if isinstance(operand, Number):
            opname, _ = UNARY_OPERATORS[op]
            return operand if opname == "+" else Number(-operand.value)
        return UnaryOp(op, operand)

    def visit_BinOp(self, node):
        left, right = self.visit(node.left), self.visit(node.right)
        if len(left.shape) == 0 or len(right.shape) == 0: pass
        elif len(left.shape) != len(right.shape):
            raise Exception("Invalid shapes: {0!s} {1!s}".format(self.dump_shape(left.shape), self.dump_shape(right.shape)))
        else:
            self.merge_shapes(left, right)
        return BinOp(type(node.op).__name__, left, right)

    def visit_BoolOp(self, node):
        values = [self.visit(value) for value in node.values]
        for value in values[1:]:
            if len(values[0].shape) != len(value.shape):
                raise Exception("Invalid shapes: {0!s} {1!s}".format(self.dump_shape(values[0].shape), self.dump_shape(value.shape)))
            else:
                self.merge_shapes(values[0], value)
        return BoolOp(type(node.op).__name__, values)

    def visit_Compare(self, node):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise UnsupportedFeatureException("More than one ops or more than one comparators are not supported")
        
        left, comparator = self.visit(node.left), self.visit(node.comparators[0])
        if len(left.shape) == 0 or len(comparator.shape) == 0: pass
        elif len(left.shape) != len(comparator.shape):
            raise Exception("Invalid shapes: {0!s} {1!s}".format(self.dump_shape(left.shape), self.dump_shape(comparator.shape)))
        else:
            self.merge_shapes(left, comparator)
        return Compare(node.ops[0], left, comparator)

    def visit_Num(self, node):
        return Number(node.n)

    def visit_Name(self, node):
        if node.id == "True":
            return Number(True)
        elif node.id == "False":
            return Number(False)
        elif not node.id in self.variables:
            if isinstance(node.ctx, ast.Store):
                variable = NewVariable(node.id)
                self.variables[node.id] = variable
                return variable
            raise Exception("Unknown variable: {0}".format(node.id))
        
        return self.variables[node.id]

    def visit_NameConstant(self, node): #new in Py34
        return Number(node.value)

    def visit_Slice(self, node):
        if not node.step is None:
            raise Exception("Step size other than 1 are not supported")
        lower = node.lower if node.lower is None else self.visit(node.lower)
        upper = node.upper if node.upper is None else self.visit(node.upper)
        if isinstance(lower, Number) and lower.value < 0:
            raise UnsupportedFeatureException("Negative slices not supported")
#         if isinstance(upper, Number) and upper.value < 0:
#             raise UnsupportedFeatureException("Negative slices not supported")
        return [(lower, upper)]

    def visit_ExtSlice(self, node):
        return [slice for dim in node.dims for slice in self.visit(dim)]

    def visit_Index(self, node):
        if isinstance(node.value, ast.Tuple):
            return [self.visit(elt) for elt in node.value.elts]
        else:
            return [self.visit(node.value)]

    def visit_If(self, node):
        return If(self.visit(node.test), self.body_visit(node.body), self.body_visit(node.orelse) if len(node.orelse) > 0 else None)

    def visit_For(self, node):
        if len(node.orelse) > 0:
            raise UnsupportedFeatureException("else conditions in forloops are not supported")
        
        iter = self.visit(node.target)
        if not isinstance(iter, NewVariable):
            raise Exception("The variable '{0}' does already exists, since the scopeing is different in c++ and python, this is not supported".format(iter.name))
        # TODO: implement this more generic
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id in ["range", "xrange"] \
                and not call_has_kw_args(node.iter):
            if len(node.iter.args) == 1:
                args = [Number(0), self.visit(node.iter.args[0])]
            elif len(node.iter.args) == 2:
                args = [self.visit(arg) for arg in node.iter.args]
            else:
                raise UnsupportedFeatureException("only forloops with a call to range(lower, upper) or range(upper) are supported")
            
            self.variables[iter.name] = Variable(iter.name, [], np.int_)
            iter = self.variables[iter.name]
            return For(iter, args[0], args[1], self.body_visit(node.body))
        else:
            raise UnsupportedFeatureException("only forloops with a call to [x]range are supported")

    def visit_While(self, node):
        if len(node.orelse) > 0:
            raise UnsupportedFeatureException("else conditions in whileloops are not supported")
        
        return While(self.visit(node.test), self.body_visit(node.body))

    def visit_Attribute(self, node):
        import hope
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and isinstance(node.value.ctx, ast.Load) \
                and node.value.id in self.fkt.__globals__ and self.fkt.__globals__[node.value.id] is hope:
            return HopeAttr(node.attr)
        elif isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and isinstance(node.value.ctx, ast.Load) \
                and node.value.id in self.fkt.__globals__ and self.fkt.__globals__[node.value.id] is np:
            if node.attr == "pi":
                return Number(np.pi)
            else:
                return NumpyAttr(node.attr)
        elif (isinstance(node.ctx, ast.Load) or isinstance(node.ctx, ast.Store)) and isinstance(node.value, ast.Name) and isinstance(node.value.ctx, ast.Load) \
                and node.value.id in self.variables and isinstance(self.variables[node.value.id], Object) \
                and (node.attr in self.variables[node.value.id].attrs or hasattr(self.variables[node.value.id].instance, node.attr)):
            return self.variables[node.value.id].getAttr(node.attr)
        elif isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Attribute):
            return self.visit(node.value).getAttr(node.attr)
        else:
            raise UnsupportedFeatureException("The attribute is not supported: {0}".format(ast.dump(node)))

    def visit_Call(self, node):
        #TODO: refactor
        # TODO: implement from bincount import bincount
        if (
                (isinstance(node.func, ast.Name) and isinstance(node.func.ctx, ast.Load))
            or (
                    isinstance(node.func, ast.Attribute) and node.func.value.id in self.variables and isinstance(self.variables[node.func.value.id], Object) \
                and (node.func.attr in self.variables[node.func.value.id].attrs or hasattr(self.variables[node.func.value.id].instance, node.func.attr)) \
                and inspect.ismethod(getattr(self.variables[node.func.value.id].instance, node.func.attr)) \
               )
        ):
            if isinstance(node.func, ast.Name) and node.func.id in self.fkt.__globals__:
                called_fkt = self.fkt.__globals__[node.func.id]
                if (inspect.isbuiltin(called_fkt) or called_fkt.__name__ == "_hope_callback") and hasattr(cache, str(id(called_fkt))):
                    called_fkt = getattr(cache, str(id(called_fkt)))
                elif not inspect.isfunction(called_fkt):
                    raise Exception("Function '{0}' not a unbound, pure python function: ({1})".format(self.fkt.__name__, ast.dump(node)))
                
                if not node.func.id in self.fkt.__globals__ or not inspect.isfunction(called_fkt):
                    raise Exception("Function '{0}' not accessible form global scope of function: ({1})".format(self.fkt.__name__, ast.dump(node)))
                
                args = [self.visit(arg) for arg in node.args]
                
            elif isinstance(node.func, ast.Attribute):
                called_fkt = getattr(self.variables[node.func.value.id].instance.__class__, node.func.attr)
                if (inspect.isbuiltin(called_fkt) or called_fkt.__name__ == "_hope_callback") and hasattr(cache, str(id(called_fkt))):
                    called_fkt = getattr(cache, str(id(called_fkt)))
                elif not inspect.isfunction(called_fkt) and not inspect.ismethod(called_fkt):
                    raise Exception("Function '{0}' not a unbound, pure python function: ({1})".format(self.fkt.__name__, ast.dump(node)))
                
                args = [self.variables[node.func.value.id]] + [self.visit(arg) for arg in node.args]
                
            else:
                raise Exception("Function '{0}' not accessible form global scope of function '{1}': ({2})".format(node.func.id, self.fkt.__name__, ast.dump(node)))
            
            if len(node.keywords) > 0:
                raise Exception("Keyword args are not allowed for global functions: {0} ({1})".format(self.fkt.__name__, ast.dump(node)))
            
            if not called_fkt.__defaults__ is None:
                raise Exception("Default args are not allowed for global functions: {0} ({1})".format(self.fkt.__name__, ast.dump(node)))
            
            if not called_fkt.__name__ in self.modtoken.functions:
                self.module_visit(called_fkt, args)
            valid = True
            if called_fkt is self.fkt:
                for arg, sig in zip(args, self.token.signature):
                    if isinstance(arg, Object) and arg.getId() != sig.getId():
                        valid = False
                    elif arg.dtype != sig.dtype or len(arg.shape) != len(sig.shape):
                        valid = False
                        
                if valid and self.token.shape is None:
                    raise Exception("Recursive call are only supported if a typed return occurs before the recursion in function: {0}".format(called_fkt.__name__))
                
                elif valid:
                    dtype, shape = self.token.dtype, self.token.shape
                    
            if not called_fkt is self.fkt or not valid:
                valid = False
                for fkt in self.modtoken.functions[called_fkt.__name__]:
                    if not valid:
                        valid = True
                        for arg, sig in zip(args, fkt.signature):
                            if isinstance(arg, Object) and arg.getId() != sig.getId():
                                valid = False
                            elif not isinstance(arg, Object) and (arg.dtype != sig.dtype or len(arg.shape) != len(sig.shape)):
                                valid = False
                                
                        if valid:
                            dtype, shape = fkt.dtype, fkt.shape
            if not valid:
                self.module_visit(called_fkt, args)
                for fkt in self.modtoken.functions[called_fkt.__name__]:
                    valid = True
                    for arg, sig in zip(args, fkt.signature):
                        if isinstance(arg, Object) and arg.getId() != sig.getId():
                            valid = False
                        elif not isinstance(arg, Object) and (arg.dtype != sig.dtype or len(arg.shape) != len(sig.shape)):
                            valid = False
                    if valid:
                        dtype, shape = fkt.dtype, fkt.shape
                        break
            if valid:
                return Call(GlobalFunction(called_fkt.__name__, [] if shape is None else shape, dtype), IterableFunctionVisitor(self.fkt.__globals__, []).visit(node), args)
            else:
                raise Exception("Unknown global function: {0}".format(node.func.id))

        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.ctx, ast.Load) \
                and isinstance(node.func.value, ast.Name) and isinstance(node.func.value.ctx, ast.Load) \
                and node.func.value.id in self.fkt.__globals__ and self.fkt.__globals__[node.func.value.id] is np \
                and node.func.attr == "sum":
            
            name, self.next = "__sum{0}".format(self.next), self.next + 1
            args, keywords = [self.visit(arg) for arg in node.args], dict([(keyword.arg, self.visit(keyword.value)) for keyword in node.keywords])
            for keyword in list(keywords.keys()):
                # TODO: support axis, keepdims
                if keyword != "dtype":
                    raise UnsupportedFeatureException("HOPE only supports 'dtype' for numpy.sum at the moment. Received: %s"%keyword)
                
            if len(args) != 1:
                raise UnsupportedFeatureException("numpy.sum requires one argument. Received: %s"%(len(args)))
            
            self.variables[name] = Variable(name, [], getattr(np, keywords["dtype"].name) if "dtype" in keywords else args[0].dtype)
            self.exprs.append(NumpyContraction(self.variables[name], "sum", args[0]))
            return self.variables[name]
            
        elif isinstance(node.func, ast.Attribute):

            if call_has_starargs(node):
                raise UnsupportedFeatureException("Only arguments without default values are supported in calls")
            if call_has_kw_args(node):
                raise UnsupportedFeatureException("Only arguments without default values are supported in calls")
            
            return Call(self.visit(node.func), 
                        IterableFunctionVisitor(self.fkt.__globals__, []).visit(node), 
                        [self.visit(arg) for arg in node.args], 
                        dict([(keyword.arg, self.visit(keyword.value)) for keyword in node.keywords])
                        )

        else:
            raise Exception("Invalid call: {0}".format(ast.dump(node)))

    def visit_Return(self, node):
        value = self.visit(node.value)
        if len(value.shape) > 0 and not isinstance(value, (Variable, View)):
            name, self.next = "__ret{0}".format(self.next), self.next + 1
            self.variables[name] = Variable(name, copy.deepcopy(value.shape), value.dtype)
            self.merge_shapes(self.variables[name], value)
            self.exprs.append(Assign(self.variables[name], value))
            value = self.variables[name]
            
        if isinstance(value, (Variable, NewVariable)):
            # ensure Py_INCREF is added when appropriate to avoid mem leak or segfault
            self.token.return_allocated = value.allocated
        
        if not self.token.dtype is None and self.token.dtype != value.dtype:
            raise Exception("None unique return type")
        
        if not self.token.shape is None and (len(self.token.shape) == 0) != (len(value.shape) == 0):
            raise Exception("Return types needs to be either all scalar or all vectors")
        
        self.token.dtype, self.token.shape = value.dtype, value.shape
        return Return(value)

    def visit_FunctionDef(self, node):
        if len(node.args.defaults) or node.args.vararg is not None or node.args.kwarg is not None:
            raise Exception("Only positional arguments with no default values are supported")
        
        if len(self.args) != len(node.args.args):
            raise Exception("Function '%s' was called with invalid number of arguments. Expected: %s, received: %s"%(node.name, len(node.args.args), len(self.args)))

        signature = self._create_signature(node)

        self.token = FunctionDef(node.name, signature)

        # removing docstring
        if (isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Str)):
            node.body.pop(0)
            
        # TODO: how do we parse recursive subfuncitons?
        self.token.body = self.body_visit(node.body) 
        AllocateVisitor(self.variables).visit(self.token.body)
        merged = {}
        for var, pos in list(self.merged.items()):
            merged[pos] = merged.get(pos, [])
            merged[pos].append(var)
        self.token.merged = list(merged.values())
        return self.token

    def body_visit(self, stmts):
        outerexprs, self.exprs, blocks, last_shape = self.exprs, [], [], "-1"
        for stmt in stmts:
            try:
                self.exprs.append(self.visit(stmt))            
            except Exception as ex:
                if "\nin body " not in (ex.args[0] if ex.args else ""):
                    from hope._tosource import tosource
                    ex.args = ((ex.args[0] if ex.args else "") + "\nin body " + tosource(stmt),) + ex.args[1:]
                raise
        for expr in self.exprs:
            segstrs = ["{0}:{1}".format("" if seg[0] is None else self.dumper.visit(seg[0]), self.dumper.visit(seg[1])) for seg in expr.shape]
            for segstr in segstrs:
                if not segstr in self.merged:
                    self.merged[segstr] = len(self.merged)
            shapestr = "/".join([repr(self.merged[segstr]) for segstr in segstrs])
            if len(shapestr) and len(blocks) > 0 and last_shape == shapestr:
                blocks[-1].body.append(expr)
            else:
                blocks.append(Block(expr))
            last_shape = shapestr
        self.exprs = outerexprs
        return Body(blocks)

    def module_visit(self, fkt, args, argid=None):
        """
        Converts the given function into a HOPE AST.
        First the source is extracted and parsed into a Python AST
        Then the AST is traversed with the arguments and a corresponding typed HOPE AST is built
        
        :param fkt: The function to be visited
        :param args: The argument, which have been passed to the function call
        :param argid: (optional) id made of the signature arguments
        """

        if argid is not None \
                and fkt.__name__ in self.modtoken.functions \
                and argid in [arg.getId() for arg in self.modtoken.functions[fkt.__name__]]:
            return

        if hasattr(self, "fkt"):
            self.stack.append((self.fkt, self.args, self.merged, self.variables, self.token))
        self.fkt, self.args, self.merged, self.variables, self.token = fkt, args, {}, {}, None

        fktast = get_fkt_ast(fkt)

        if len(fktast.args.defaults) or fktast.args.vararg is not None or fktast.args.kwarg is not None:
            raise Exception("Only positional arguments with no default values are supported")

        if not fkt.__name__ in self.modtoken.functions:
            self.modtoken.functions[fkt.__name__] = []

        if argid is None or not argid in self.modtoken.functions[fkt.__name__]:
            #Create the HOPE AST from the builtin AST
            parsed = self.visit(fktast)
            if not parsed.getId() in [arg.getId() for arg in self.modtoken.functions[fkt.__name__]]:
                self.modtoken.functions[fkt.__name__].append(parsed)

        if len(self.stack):
            self.fkt, self.args, self.merged, self.variables, self.token = self.stack.pop()
        else:
            delattr(self, "args"), delattr(self, "fkt")

    def _create_signature(self, node):
        """
        Creates a method signature for the given :py:class:`hope._ast.FunctionDef`
        
        :param node: instance of the FunctionDef
        
        :return signature: a list of :py:class:`hope._ast.Variable`
        """
        signature = []
        for name, arg in zip(node.args.args, self.args):
            if sys.version_info[0] == 2 and isinstance(name, ast.Name):
                if not isinstance(name.ctx, ast.Param):
                    raise Exception("Invalid Structure")
                
                argname = name.id
            elif sys.version_info[0] == 3 and isinstance(name, ast.arg):
                if not name.annotation is None:
                    raise Exception("Invalid Structure")
                
                argname = name.arg
            else:
                raise Exception("Invalid Structure")
            
            if isinstance(arg, (Variable, Object)):
                #TODO: maybe implement prototype pattern. 
                #Overwrite attributes of passed arguments to avoid conflicts
                self.variables[argname] = copy.copy(arg)
                self.variables[argname].name = argname
                self.variables[argname].scope = "signature"
                
            elif isinstance(arg, np.ndarray):
                if arg.dtype.isbuiltin != 1:
                    raise UnsupportedFeatureException("Only Builtin datatypes are supported: in {0} arg {1} has type {2!s}".format(self.fkt.__name__, argname, arg.dtype))
                
                self.variables[argname] = Variable(argname, None, arg.dtype.type, "signature", True)
                self.variables[argname].shape = [(None, Dimension(self.variables[argname], dim)) for dim in range(len(arg.shape))]
            elif hasattr(arg, "dtype"):
                if hasattr(arg.dtype, "type"):
                    self.variables[argname] = Variable(argname, [], arg.dtype.type, "signature", True)
                else:
                    self.variables[argname] = Variable(argname, [], arg.dtype, "signature", True)
            elif isinstance(arg, int):
                self.variables[argname] = Variable(argname, [], int, "signature", True)
            elif isinstance(arg, float):
                self.variables[argname] = Variable(argname, [], float, "signature", True)
            else:
                self.variables[argname] = Object(argname, arg)
            signature.append(self.variables[argname])
            
        return signature

def get_fkt_ast(fkt):
    """
    Creates a AST form the given function by getting the source of the function 
    by inspection, removing unneccesary indentations and passing the string 
    representation to the :py:mod:`ast` module. 
    
    :param fkt: Function object to be used
    
    :return fktast: the AST representation of the function
    
    :raises Exception: if the function could not be processed
    """
    lines = inspect.getsource(fkt).split("\n")
    for line in lines:
        if line.find("def") > -1 and (line.find("#") == -1 or line.find("#") > line.find("def")):
            outdent = line.find("def")
            break
    source = "\n".join([line[outdent:] for line in lines])
    modast = ast.parse(source)

    if not isinstance(modast, ast.Module) or len(modast.body) != 1 or not isinstance(modast.body[0], ast.FunctionDef):
        raise Exception("Error decompiling function")
    fktast = modast.body[0]
    return fktast
