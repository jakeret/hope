# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

from __future__ import print_function, division, absolute_import, unicode_literals


from hope._ast import *


class Dumper(NodeVisitor):
    """
    Generates a human readable string representation of a HOPE AST.
    Can be used to dump an AST for debugging purpose. Is internally
    used to generate consistent variable names etc. 
    """

    def visit_Number(self, node):
        return "{0!r}.{1}".format(node.value, PY_TYPE_CHAR[node.dtype])

    def visit_NewVariable(self, node):
        return "{0}.new".format(node.name)

    def visit_Variable(self, node):
        ret = node.name
        if not node.dtype is None:
            ret += ".{0}".format(PY_TYPE_CHAR[node.dtype])
        if len(node.shape) > 0:
            segments = []
            for lower, upper in node.shape:
                segments.append("{0}:{1}".format("" if lower is None else "{0!s}".format(lower), self.visit(upper)))
            ret += "[{0}]".format(",".join(segments))
        return ret

    def visit_Object(self, node):
        return "{0}{1}@{2}".format(("{0}->".format(self.visit(node.parent)) if node.parent is not None else ""), node.name, node.classname)

    def visit_ObjectAttr(self, node):
        ret = "{0}->{1}".format(self.visit(node.parent), node.name)
        if not node.dtype is None:
            ret += ".{0}".format(PY_TYPE_CHAR[node.dtype])
        if len(node.shape) > 0:
            segments = []
            for lower, upper in node.shape:
                segments.append("{0}:{1}".format("" if lower is None else "{0!s}".format(lower), self.visit(upper)))
            ret += "[{0}]".format(",".join(segments))
        return ret

    def visit_Dimension(self, node):
        if isinstance(node.variable, ObjectAttr):
            parent = node.variable.parent
            trace = [node.variable.name]
            while not parent is None:
                trace.insert(0, parent.name)
                parent = parent.parent
            return "{0}@{1}".format(".".join(trace), node.dim)
        elif isinstance(node.variable, Variable):
            return "{0}@{1}".format(node.variable.name, node.dim)
        else:
            raise Exception("Unknown type {0}".format(node.variable.name))
        
    def visit_DimensionSlice(self, node):
        return "{0}+{1}".format(self.visit_Dimension(node), self.visit(node.slice))

    def visit_View(self, node):
        ret = node.variable.name
        if not node.variable.dtype is None:
            ret += ".{0}".format(PY_TYPE_CHAR[node.variable.dtype])
        shape = []
        for variable_extent, extent in zip(node.variable.shape, node.extents):
            if isinstance(extent, tuple):
                lower, upper = extent
                if lower is None: lower = variable_extent[0]
                if isinstance(lower, Number) and lower.value == 0: lower = None
                if upper is None: upper = variable_extent[1]
                shape.append("{0}:{1}".format("" if lower is None else self.visit(lower), self.visit(upper)))
            else:
                shape.append(self.visit(extent))
        ret += "[{0}]".format(", ".join(shape))
        return ret

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Assign(self, node):
        return "{0} = {1}".format(self.visit(node.target), self.visit(node.value))

    def visit_AugAssign(self, node):
        return "{0} {1} {2}".format(self.visit(node.target), node.op, self.visit(node.value))

    def visit_UnaryOp(self, node):
        return "{0}{1}".format(node.op, self.visit(node.operand))

    def visit_BinOp(self, node):
        return "({0} {1} {2})".format(self.visit(node.left), node.op, self.visit(node.right))

    def visit_BoolOp(self, node):
        return "({0})".format(" {0} ".format(node.op).join([self.visit(value) for value in node.values]))

    def visit_Compare(self, node):
        return "({0} {1} {2})".format(self.visit(node.left), node.op, self.visit(node.comparator))

    def visit_If(self, node):
        ret = "if {0} {{\n\t{1}\n}}".format(self.visit(node.test), "\n\t".join(self.visit(node.body).split("\n")))
        if not node.orelse is None:
            ret += " else {{\n\t{0}\n}}".format("\n\t".join(self.visit(node.orelse).split("\n")))
        return ret

    def visit_For(self, node):
        return "for {0} in ({1}:{2}) {{\n\t{3}\n}}".format(self.visit(node.iter), self.visit(node.lower), self.visit(node.upper), "\n\t".join(self.visit(node.body).split("\n")))

    def visit_While(self, node):
        return "while {0} {{\n\t{1}\n}}".format(self.visit(node.test), "\n\t".join(self.visit(node.body).split("\n")))

    def visit_Call(self, node):
        keywords = ["{0}={1}".format(name, self.visit(value)) for name, value in list(node.keywords.items())]
        return "{0}({1})".format(self.visit(node.name), ", ".join([self.visit(arg) for arg in node.args] + keywords))

    def visit_GlobalFunction(self, node):
        return node.name

    def visit_HopeAttr(self, node):
        return "hope.{0}".format(node.name)

    def visit_NumpyAttr(self, node):
        return "numpy.{0}".format(node.name)

    def visit_NumpyContraction(self, node):
        return "{0} = numpy.{1}({2})".format(self.visit(node.variable), node.op, self.visit(node.value))

    def visit_Allocate(self, node):
        return "new {0}".format(self.visit(node.variable))

    def visit_Return(self, node):
        return "return {0}".format(self.visit(node.value))

    def visit_Block(self, node):
        if len(node.shape):
            segments = ""
            for lower, upper in node.shape:
                segments += "({0}:{1})".format("" if lower is None else self.visit(lower), self.visit(upper))
            return "{0} {{\n\t{1}\n}}".format(segments, "\n\t".join("\n".join([self.visit(expr) for expr in node.body]).split("\n")))
        else:
            return "\n".join([self.visit(expr) for expr in node.body])

    def visit_Body(self, node):
        return "\n".join([self.visit(block) for block in node.blocks])

    def visit_FunctionDef(self, node):
        args = []
        for arg in node.signature:
            if isinstance(arg, Variable):
                args.append("{0}{1!s} {2}".format(NPY_TYPE[arg.dtype], "" if len(arg.shape) == 0 else "^{0}".format(len(arg.shape)), arg.name))
            elif isinstance(arg, Object):
                args.append("{0} {1}".format(arg.classname, arg.name))
            else:
                raise Exception("Unknown signature argument: {0!s}".format(self.visit(arg)))
        return "{0}({1})".format(node.name, ", ".join(args)) + "\n\t" + ("\n\t".join(self.visit(node.body).split("\n"))) + "\n"

    def visit_Module(self, node):
        code = ""
        for fktcls in list(node.functions.values()):
            for fkt in fktcls:
                code += self.visit(fkt)
        return code
