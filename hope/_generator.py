# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

from __future__ import print_function, division, absolute_import, unicode_literals


import sys
import pickle

from hope import config
from hope._ast import *
from hope._const import *
from hope._library import *
from hope._dump import Dumper

class CPPGenerator(NodeVisitor):
    """
    Generates the C code from the given :py:class:`hope._ast.Module` token
    by traversing the AST.
    """

    def __init__(self):
        self.next_loopid, self.merged, self.slicemap, self.library, self.dumper  = 0, None, {}, {}, Dumper()

    def getVariableExtent(self, node):
        extent = ""
        for ind, segment in enumerate(node.shape):
            segmentstr = self.get_segmentstr(*segment)
            if not segment[0] is None:
                raise Exception("Variable slices needs to start with None: {0}".format(node.name))
            if not segmentstr in self.merged:
                raise Exception("Unknown slice {0} in variable {1}".format(segmentstr, node.name))
            if ind > 0:
                extent = "({0})*{1}".format(extent, self.visit(segment[1]))
            extent += "{0}{1}".format(" + " if len(extent) > 0 else "", self.slicemap[self.get_slicemap_key(ind, *segment)])
        return extent

    def visit_Number(self, node):
        if node.dtype is bool:
            return "true" if node.value else "false"
        if config._readablecxx:
            return "{1!r}".format(PY_C_TYPE[node.dtype], node.value)
        else:
            return "({0}){1!r}".format(PY_C_TYPE[node.dtype], node.value)

    def visit_Variable(self, node):
        if len(node.shape) == 0 or node.scope == "block":
            return "c{0}".format(node.name)
        else:
            return "c{0}[{1}]".format(node.name, self.getVariableExtent(node));

    def visit_ObjectAttr(self, node):
        trace = node.getTrace()
        if len(node.shape) == 0:
            return "c" + ".c".join(trace)
        else:
            return "{0}[{1}]".format("c" + ".c".join(trace), self.getVariableExtent(node));

    def visit_Dimension(self, node):
        if isinstance(node.variable, ObjectAttr):
            parent = node.variable.parent
            trace = ["s{0}".format(node.variable.name)]
            while not parent is None:
                trace.insert(0, "c{0}".format(parent.name))
                parent = parent.parent
            return "{0}[{1}]".format(".".join(trace), node.dim)
        elif isinstance(node.variable, Variable):
            return "s{0}[{1}]".format(node.variable.name, node.dim)
        else:
            raise Exception("Unknown type {0}".format(node.variable.name))

    def visit_DimensionSlice(self, node):
        return "{0}+{1}".format(self.visit_Dimension(node), self.visit(node.slice))

    def visit_View(self, node):
        subscript = ""
        extent_ind = 0
        for ind, (extent, segment) in enumerate(zip(node.extents, node.variable.shape)):
            if ind > 0:
                subscript = "(int)({0})*{1}".format(subscript, self.visit(segment[1]))
            subscript += " + " if len(subscript) > 0 else ""
            if isinstance(extent, tuple):
                lower, upper = extent
                if lower is None: lower = segment[0]
                if isinstance(lower, Number) and lower.value == 0: lower = None
                if upper is None: upper = segment[1]
                seg = "{0} + ".format(self.visit(lower)) if not lower is None else ""
                key = self.get_slicemap_key(extent_ind, lower, upper)
                seg += self.slicemap[key]
#                 seg += self.slicemap[self.merged["{0}:{1}".format("" if lower is None else self.dumper.visit(lower), self.dumper.visit(upper))]]
                segstr = "{0}:{1}".format("" if lower is None else self.dumper.visit(lower), self.dumper.visit(upper))
                extent_ind += 1
            else:
                seg = self.visit(extent)
                if isinstance(extent, Number) and extent.value < 0:
                    seg = "{0}+{1}".format(self.visit_Dimension(segment[1]), seg)
                segstr = self.dumper.visit(extent)
            if config.rangecheck:
                subscript += "native_rangecheck({0}".format(seg)
                subscript += ", {0}, {1}".format(self.visit(segment[0]) if not segment[0] is None else "0", self.visit(segment[1]))
                subscript += ", std::string(\"{0}\"), std::string(\"{1}\"))".format(segstr, node.variable.name)
                self.library["native_rangecheck"] = LIBRARY_NATIVE_RANGECHECK
            else:
                subscript += seg
        shape = node.variable.shape
        node.variable.shape = []
        # TODO: if double, cast to int ...
        ret = "{0}[(int)({1})]".format(self.visit(node.variable), subscript);
        node.variable.shape = shape
        return ret

    def visit_Expr(self, node):
        return "{0};".format(self.visit(node.value))

    def visit_NumpyContraction(self, node):
        if node.op != "sum":
            raise Exception("Only the numpy.sum contraction is implemented!")
        if len(node.value.shape):
            ret = "{0} = ({1})0;\n".format(self.visit(node.variable), PY_C_TYPE[node.dtype])
            keys = []
            for ind, segment in enumerate(node.value.shape):
                ret += "{0}for (npy_intp i{1} = 0; i{1} < {2} - {3}; ++i{1}) {{\n".format( \
                      "\t" * ind \
                    , self.next_loopid \
                    , self.visit(segment[1]) \
                    , 0 if segment[0] is None else self.visit(segment[0]) \
                )
                segmentstr = self.get_segmentstr(*segment)
                if not segmentstr in self.merged:
                    self.merged[segmentstr] = segmentstr
                keys.append(self.get_slicemap_key(ind, *segment))
                self.slicemap[keys[-1]] = "i{0}".format(self.next_loopid)
                self.next_loopid += 1
            ret += "{0}{1} += {2};".format("\t" * len(node.value.shape), self.visit(node.variable), self.visit(node.value))
            for ind, (key, segment) in enumerate(zip(keys, node.value.shape)):
                del self.slicemap[key]
                ret += "\n{0}}}".format("\t" * (len(node.value.shape) - 1 - ind))
            return ret
        else:
            return "\n{0} = {1};".format(self.visit(node.variable), self.visit(node.value))

    def visit_Assign(self, node):
        # TODO: capture type
        if isinstance(node.target, Variable) and node.target.allocated == False:
            node.target.allocated = True
            return "auto {0} = {1};".format(self.visit(node.target), self.visit(node.value))
        else:
            return "{0} = {1};".format(self.visit(node.target), self.visit(node.value))

    def visit_Reference(self, node):
        target = node.target
        trace = node.value.getTrace()
        if isinstance(target, ObjectAttr): # self.x = self.y
            return "c{0} = c{1};".format(".c".join(target.getTrace()), ".c".join(trace))
        if len(target.shape) == 0: # [int] x = self.y
            return "{0} c{1} = c{2};".format(PY_C_TYPE[target.dtype], target.name, ".c".join(trace))
        else: # [array] x = self.y
            return "PyObject * p{0} = (PyObject *)PyArray_GETCONTIGUOUS((PyArrayObject *)c{1});\n".format(target.name, ".p".join(trace)) \
                +  "npy_intp * s{0} = c{1};\n".format(target.name, ".s".join(trace)) \
                +  "{0} * c{1} = c{2};".format(PY_C_TYPE[target.dtype], target.name, ".c".join(trace))

    def visit_AugAssign(self, node):
        if node.op == "**=":
            return "{0} = std::pow({0}, {1});".format(self.visit(node.target), self.visit(node.value))
        elif node.op == "//=":
            if type(node.target.dtype(1) // node.value.dtype(1)) in [float, np.float32, np.float64]:
                return "{0} = std::floor({0} / {1});".format(self.visit(node.target), self.visit(node.value))
            else:
                return "{0} /= {1};".format(self.visit(node.target), self.visit(node.value))
        elif node.op == "%=":
            self.library["native_mod"] = LIBRARY_NATIVE_MOD
            return "{0} = native_mod({0}, {1});".format(self.visit(node.target), self.visit(node.value))
        else:
            return "{0} {1} {2};".format(self.visit(node.target), node.op, self.visit(node.value))

    def visit_UnaryOp(self, node):
        if config._readablecxx:
            return "{0}{1}".format(node.op, self.visit(node.operand))
        else:
            return "({0}{1})".format(node.op, self.visit(node.operand))

    def visit_BinOp(self, node):
        cast = "" if node.dtype is None or config._readablecxx else "({0})".format(PY_C_TYPE[node.dtype])
        if node.op == "**":
            return "{0}std::pow({1}, {2})".format(cast, self.visit(node.left), self.visit(node.right))
        elif node.op == "//" and node.dtype in [float, np.float32, np.float64]:
            return "{0}std::floor({1} / {2})".format(cast, self.visit(node.left), self.visit(node.right))
        elif node.op == "//":
            return "{0}({1} / {2})".format(cast, self.visit(node.left), self.visit(node.right))
        else:
            left = self.visit(node.left) if node.dtype == node.left else "{0}{1}".format(cast, self.visit(node.left))
            right = self.visit(node.right) if node.dtype == node.right else "{0}{1}".format(cast, self.visit(node.right))
            if node.op == "%":
                self.library["native_mod"] = LIBRARY_NATIVE_MOD
                return "native_mod({0}, {1})".format(left, right)
            else:
                return "{0}({1} {2} {3})".format(cast, left, node.op, right)

    def visit_BoolOp(self, node):
        values = [self.visit(value) for value in node.values]
        return "({0})".format(" {0} ".format(node.op).join(values))

    def visit_Compare(self, node):
        cast = "" if node.dtype is None or config._readablecxx else "({0})".format(PY_C_TYPE[node.dtype])
        return "{0}({1} {2} {3})".format(cast, self.visit(node.left), node.op, self.visit(node.comparator))

    def visit_If(self, node):
        ret = "if ({0}) {{\n\t{1}\n}}".format(self.visit(node.test), "\n\t".join(self.visit(node.body).split("\n")))
        if not node.orelse is None:
            ret += " else {{\n\t{0}\n}}".format("\n\t".join(self.visit(node.orelse).split("\n")))
        return ret

    def visit_For(self, node):
        ret = "for (npy_intp {0} = {1}; {0} < {2}; ++{0}) {{\n\t".format(self.visit(node.iter), self.visit(node.lower), self.visit(node.upper))
        ret += "\n\t".join(self.visit(node.body).split("\n"))
        ret += "\n}"
        return ret

    def visit_While(self, node):
        ret = "while ({0}) {{\n\t".format(self.visit(node.test))
        ret += "\n\t".join(self.visit(node.body).split("\n"))
        ret += "\n}"
        return ret

    def visit_Call(self, node):
        # TODO: generalize
        if isinstance(node.name, GlobalFunction):
            args = []
            for arg in node.args:
                if not isinstance(arg, Object) and len(arg.shape) > 0 and (not isinstance(arg, Variable) or arg.scope == "block"):
                    raise Exception("Only variables can be passed to funtions!")
                elif isinstance(arg, Object):
                    args.append("c{0}".format(arg.name))
                elif len(arg.shape) > 0:
                    args.append("p{0}, s{0}, c{0}".format(arg.name))
                else:
                    args.append(self.visit(arg))
            return "{0}_{1}({2})".format( \
                  node.name.name \
                , "".join([arg.getId() for arg in node.args]) \
                , ", ".join(args) \
            )
        elif isinstance(node.name, HopeAttr) and node.name.name == "exp":
            self.library["hope_exp"] = LIBRARY_HOPE_EXP
            return "hope_exp({0})".format(self.visit(node.args[0]))
        elif isinstance(node.name, NumpyAttr) and node.name.name in ["empty", "zeros", "ones"]:
            return "1" if node.name.name == "ones" else "0"
        elif isinstance(node.name, NumpyAttr) and node.name.name == "interp":
            self.library["numpy_interp"] = LIBRARY_NUMPY_INTERP

            args = [[] for _ in range(3)]
            for i in range(1, 3):
                if isinstance(node.args[i], ObjectAttr):
                    parent, trace = node.args[i].parent, [node.args[i].name]
                    while not parent is None:
                        trace.insert(0, parent.name)
                        parent = parent.parent
                    args[i] = ".c".join(trace)
                else:
                    args[i] = node.args[i].name
                    
            # TODO: make sure node.args[1].shape == node.args[2].shape using an assert
            
            ((lower, upper),) = node.args[1].shape
            left_val = "c{0}[0]".format(node.args[1].name) if lower is None else self.visit(lower)
            size = self.visit(upper) + ("" if lower is None else "-{0}".format(self.visit(lower)))
            right_val = "c{0}[{1}]".format(node.args[1].name, size + "-1")

            ret = "numpy_interp({0}, c{1}, c{2}, {3})".format(self.visit(node.args[0]), args[1], args[2], size)
            
            if "left" in node.keywords:
                left_ret = self.visit(node.keywords["left"])
            else:
                left_ret = "c{0}[0]".format(args[2])
                
            ret = "{0} < {1} ? {2} : ({3})".format(self.visit(node.args[0]), left_val, left_ret, ret)
                
            if "right" in node.keywords:
                right_ret = self.visit(node.keywords["right"])
            else:
                right_ret = "c{0}[{1}]".format(args[2], size + "-1")
                
            ret = "{0} > {1} ? {2} : ({3})".format(self.visit(node.args[0]), right_val, right_ret, ret)
                
            return "({0})".format(ret)
        elif isinstance(node.name, NumpyAttr) and node.name.name == "sign":
            self.library["native_sign"] = LIBRARY_NATIVE_SIGN
            return "native_sign({0})".format(self.visit(node.args[0]))
        elif isinstance(node.name, NumpyAttr) and node.name.name in NPY_UNARY_FUNCTIONS:
            return "{0}({1})".format(NPY_UNARY_FUNCTIONS[node.name.name], self.visit(node.args[0]))
        elif isinstance(node.name, NumpyAttr) and node.name.name in NPY_CAST_FUNCTIONS:
            return "({0})({1})".format(PY_C_TYPE[NPY_CAST_FUNCTIONS[node.name.name]], self.visit(node.args[0]))

    def visit_Allocate(self, node):
        shape, variable = [], node.variable
        if variable.dtype is None:
            raise Exception("Unknown dtype: {0}".format(variable.dtype))
        for segment in variable.shape:
            if not segment[0] is None:
                raise Exception("Allocate need to have (:len)* in shape: {0}".format(",".join([str(sgment) for sgment in variable.shape])))
            shape.append(self.visit(segment[1]))
        if len(shape) == 0:
            return "{0} c{1} = {0}();".format(PY_C_TYPE[variable.dtype], variable.name)
        else:
            return "npy_intp d{0}[] = {{(npy_intp){1}}};\n".format(variable.name, ", (npy_intp)".join(shape)) \
                +  "PyObject * p{0} = PyArray_EMPTY({1}, d{0}, {2}, 0);\n".format(variable.name, len(shape), NPY_TYPEENUM[variable.dtype]) \
                +  "npy_intp * s{0} = PyArray_SHAPE((PyArrayObject *)p{0});\n".format(variable.name) \
                +  "{0} * c{1} = ({0} *)PyArray_DATA((PyArrayObject *)p{1});".format(PY_C_TYPE[variable.dtype], variable.name)

    def visit_Return(self, node):
        # TODO: implement expressions
        if len(node.value.shape) == 0:
            return "return {0};".format(self.visit(node.value));
        elif not isinstance(node.value, Variable):
            raise Exception("TODO: implement!")
        else:
            return "return std::make_tuple((PyObject *)p{0}, s{0}, c{0});".format(node.value.name);

    def visit_Block(self, node):
        if len(node.shape):
            ret = "";
            keys = []
            for ind, segment in enumerate(node.shape):
                ret += "{0}for (npy_intp i{1} = 0; i{1} < {2} - {3}; ++i{1}) {{\n".format(
                          "\t" * ind, 
                          self.next_loopid, 
                          self.visit(segment[1]), 
                          0 if segment[0] is None else self.visit(segment[0])
                        )
                keys.append(self.get_slicemap_key(ind, *segment))
                self.slicemap[keys[-1]] = "i{0}".format(self.next_loopid)
                self.next_loopid += 1
                
            ret += "{0}".format("\t" * len(node.shape))
            ret += "\n{0}".format("\t" * len(node.shape)).join("\n".join([self.visit(expr) for expr in node.body]).split("\n"))
            
            for ind, key in enumerate(keys):
                del self.slicemap[key]
                ret += "\n{0}}}".format("\t" * (len(node.shape) - 1 - ind))
                
            return ret
        else:
            return "\n".join(["{0}".format(self.visit(expr)) for expr in node.body])

    def visit_Body(self, node):
        return "\n".join([self.visit(block) for block in node.blocks])

    def visit_FunctionDef(self, node):
        self.merged, code, firstSegment = {}, "", None
        for merged in node.merged:
            if len(merged) == 1:
                self.merged[merged[0]] = merged[0]
            else:
                for segment in merged:
                    self.merged[segment] = merged[0]
                    if isinstance(node.shapes[segment], tuple):
                        lower, upper = node.shapes[segment]
                        if isinstance(lower, Number) and lower.value == 0: lower = None
                        if upper is None:
                            raise Exception("Unbound shapes cannot be merged: {0}".format(segment))
                        if lower is None:
                            shape = self.visit(upper)
                        else:
                            shape = "({1} - {0})".format(self.visit(lower), self.visit(upper))
                    else:
                        raise Exception("Indexes can not be merged: {0!s}".format(segment))
                    if firstSegment is None:
                        firstSegment = shape
                    # TODO: this fails if variable is defined in loop, maybe it makes sense to visit lower and upper und check if we can check ...
                    elif config.rangecheck:
                        code += "\n\tif ({0} - {1} != 0) {{".format(firstSegment, shape)
                        code += "\n\t\tPyErr_SetString(PyExc_ValueError, \"Shapes {0} and {1} do not match!\");".format(firstSegment, shape)
                        code += "\n\t\tthrow std::exception();"
                        code += "\n\t}"
        code += "\n\t" + "\n\t".join(self.visit(node.body).split("\n"))
        if not node.dtype is None:
            code += "\n\tPyErr_SetString(PyExc_ValueError, \"No return type passed!\");"
            code += "\n\tthrow std::exception();"
        return code

    def visit_Module(self, node):
        for fktname, fktlist in list(node.functions.items()):
            for fkt in fktlist:
                fkt.decl, sig = "inline ", []
                if fkt.dtype is None:
                    fkt.decl += "void"
                elif len(fkt.shape) == 0:
                    fkt.decl += PY_C_TYPE[fkt.dtype]
                else:
                    fkt.decl += "std::tuple<PyObject *, npy_intp const *, {0} *>".format(PY_C_TYPE[fkt.dtype])
                fkt.decl += " {0}_{1}(\n\t  ".format(fktname, fkt.getId())
                for arg in fkt.signature:
                    if isinstance(arg, Object):
                        sig.append("{0} & c{1}\n".format(arg.getId("t"), arg.name))
                    elif len(arg.shape) > 0:
                        sig.append("PyObject * p{1}, npy_intp const * __restrict__ s{1}, {0} * __restrict__ c{1}\n".format(PY_C_TYPE[arg.dtype], arg.name))
                    else:
                        sig.append("{0} c{1}\n".format(PY_C_TYPE[arg.dtype], arg.name))
                fkt.decl += "\t, ".join(sig) + ")"
        code = "".join([fkt.decl + ";\n" for fktname, fktlist in list(node.functions.items()) for fkt in fktlist])

        for fktname, fktlist in list(node.functions.items()):
            for fkt in fktlist:
                code += fkt.decl + " {"
                code += self.visit(fkt)
                code += "\n}\n"
        return "\n".join(list(self.library.values())) + code


    def get_segmentstr(self, lower, upper):
        return "{0}:{1}".format("" if lower is None else self.dumper.visit(lower), self.dumper.visit(upper))
    
    def get_slicemap_key(self, ind, lower, upper):
        segmentstr = self.get_segmentstr(lower, upper)
        return "i{0}>{1}".format(ind, self.merged[segmentstr])

def generate(modtoken, localfilename):
    """
    Generates the C code from the given :py:class:`hope._ast.Module` token
    
    :param modtoken: Module to use
    :param localfilename: name of the function incl. signature
    
    :return code: the generated C code
    """

    objects = []
    def findObjects(obj):
        if obj.getId() not in [arg.getId() for arg in objects]:
            objects.insert(0, obj)
        for variable in list(obj.attrs.values()):
            if isinstance(variable, Object):
                findObjects(variable)

    for fktlist in list(modtoken.functions.values()):
        for fkt in fktlist:
            for arg in fkt.signature:
                if isinstance(arg, Object):
                    findObjects(arg)

    code  = LIBRARY_IMPORTS
    code += LIBRARY_PYOBJ_DEF
    
    code += _obj_init_code(objects)
    
    generator = CPPGenerator()
    code += generator.visit(copy.deepcopy(modtoken))

    code += LIBRARY_SIGHANDLER

    code += "\n"
    code += "extern \"C\" {\n"
    
    code += LIBRARY_CREATE_SIGNATURE

    code += _run_fkt_code(modtoken)
    
    code += "\t}\n"
    #end of extern block

    if sys.version_info[0] == 2:
        code += LIBRARY_METHODS_DECL_PY2.format(fktname=modtoken.main)
        code += LIBRARY_INIT_DECL_PY2.format(filename=localfilename, fktname=modtoken.main)
        
    else:
        code += LIBRARY_METHODS_DECL_PY3.format(fktname=modtoken.main)
        code += LIBRARY_MODULE_DECL_PY3.format(fktname=modtoken.main)
        code += LIBRARY_INIT_DECL_PY3.format(filename=localfilename, fktname=modtoken.main)

    code += "}\n"

    return code

def _obj_init_code(objects):
    code = ""
    for obj in objects:

        code += "struct {0} {{\n".format(obj.getId("t"))

        code += "\tbool initialize(PyObject * obj) {"
        if len(obj.attrs) == 0:
            code += "\n\t\treturn true;"
        else:
            code += "\n\t\tif ("
            for pos, (name, variable) in enumerate(obj.attrs.items()):
                code += "\n\t\t\t"
                if pos > 0:
                    code += "and "
                code += "PyObject_HasAttrString(obj, \"{0}\") and p{0}.incref(PyObject_GetAttrString(obj, \"{0}\")) ".format(name)
                if isinstance(variable, Object):
                    code += "and c{0}.initialize(p{0})".format(name)
                elif len(variable.shape) > 0:
                    code += "and PyArray_CheckExact(p{0})".format(name)
                    code += "\n\t\t\tand PyArray_TYPE((PyArrayObject *)p{0}) == {1} and PyArray_NDIM((PyArrayObject *)p{0}) == {2}".format(name, NPY_TYPEENUM[variable.dtype], len(variable.shape))
                elif variable.dtype is int:
                    if sys.version_info[0] == 2:
                        code += "and PyInt_CheckExact((PyObject *)p{0})".format(name)
                    else:
                        code += "and PyLong_CheckExact((PyObject *)p{0})".format(name)
                elif variable.dtype is float:
                    code += "and PyFloat_CheckExact((PyObject *)p{0})".format(name)
                elif variable.dtype in NPY_SCALAR_TAG:
                    code += "and PyArray_IsScalar((PyArrayObject *)p{0}, {1})".format(name, NPY_SCALAR_TAG[variable.dtype])
                else:
                    raise Exception("Unknown type: {0!s}".format(variable.dtype))
            code += "\n\t\t) {\n"
            for name, variable in list(obj.attrs.items()):
                if isinstance(variable, Object):
                    pass
                elif len(variable.shape) > 0:
                    code += "\t\t\tif (!(p{0}.incref((PyObject *)PyArray_GETCONTIGUOUS((PyArrayObject *)p{0})))) {{\n".format(name)
                    code += "\t\t\t\tPyErr_SetString(PyExc_ValueError, \"Invalid Argument type on {0}!\");\n".format(name)
                    code += "\t\t\t\tthrow std::exception();\n"
                    code += "\t\t\t}\n"
                    code += "\t\t\ts{0} = PyArray_SHAPE((PyArrayObject *)p{0});\n".format(name)
                    code += "\t\t\tc{1} = ({0} *)PyArray_DATA((PyArrayObject *)p{1});\n".format(PY_C_TYPE[variable.dtype], name)
                elif variable.dtype is int:
                    if sys.version_info[0] == 2:
                        code += "\t\t\tc{0} = PyInt_AS_LONG((PyObject *)p{0});\n".format(name)
                    else:
                        code += "\t\t\tc{0} = PyLong_AS_LONG((PyObject *)p{0});\n".format(name)
                elif variable.dtype is float:
                    code += "\t\t\tc{0} = PyFloat_AS_DOUBLE((PyObject *)p{0});\n".format(name)
                elif variable.dtype in NPY_SCALAR_TAG:
                    code += "\t\t\tc{0} = PyArrayScalar_VAL((PyArrayObject *)p{0}, {1});\n".format(name, NPY_SCALAR_TAG[variable.dtype])
            code += "\t\t\treturn true;\n"
            code += "\t\t} else\n"
            code += "\t\t\treturn false;\n"

        code += "\t}\n"
        for name, variable in list(obj.attrs.items()):
            if isinstance(variable, Object):
                code += "\tPyObj p{1};\n\t{0} c{1};\n".format(variable.getId("t"), name)
            elif len(variable.shape) > 0:
                code += "\tPyObj p{1};\n\tnpy_intp * s{1};\n\t{0} * __restrict__ c{1};\n".format(PY_C_TYPE[variable.dtype], name)
            else:
                code += "\tPyObj p{1};\n\t{0} c{1};\n".format(PY_C_TYPE[variable.dtype], name)
        code += "};\n"

    return code

def _run_fkt_code(modtoken):
    code = ""
    code += "\tPyObject * run(PyObject * self, PyObject * args) {\n"

    for fkt in modtoken.functions[modtoken.main]:
        code += "\t\t{"
        for arg in fkt.signature:
            if isinstance(arg, Object):
                code += "\n\t\t\tPyObject * p{0};".format(arg.name)
                code += "\n\t\t\t{0} c{1};".format(arg.getId("t"), arg.name)
            elif len(arg.shape) > 0:
                code += "\n\t\t\tPyObj p{0};".format(arg.name)
            else:
                code += "\n\t\t\tPyObject * p{0};".format(arg.name)
                code += " {0} c{1};".format(PY_C_TYPE[arg.dtype], arg.name)
        if len(fkt.signature) > 0:
            code += "\n\t\t\tif ("
            code += "\n\t\t\t\tPyTuple_CheckExact(args) and PyTuple_GET_SIZE(args) == {0}".format(len(fkt.signature))
            for idx, arg in enumerate(fkt.signature):
                code += "\n\t\t\t\tand (p{0} = PyTuple_GET_ITEM(args, {1})) ".format(arg.name, idx)
                if isinstance(arg, Object):
                    code += "and c{0}.initialize(p{0})".format(arg.name)
                elif len(arg.shape) > 0:
                    code += "and PyArray_CheckExact(p{0})".format(arg.name)
                    code += "\n\t\t\t\tand PyArray_TYPE((PyArrayObject *)p{0}) == {1} and PyArray_NDIM((PyArrayObject *)p{0}) == {2}".format(arg.name, NPY_TYPEENUM[arg.dtype], len(arg.shape))
                elif arg.dtype is int:
                    if sys.version_info[0] == 2:
                        code += "and PyInt_CheckExact(p{0})".format(arg.name)
                    else:
                        code += "and PyLong_CheckExact(p{0})".format(arg.name)
                elif arg.dtype is float:
                    code += "and PyFloat_CheckExact(p{0})".format(arg.name)
                elif arg.dtype in NPY_SCALAR_TAG:
                    code += "and PyArray_IsScalar(p{0}, {1})".format(arg.name, NPY_SCALAR_TAG[arg.dtype])
                else:
                    raise Exception("Unknown type: {0!s}".format(arg.dtype))

            code += "\n\t\t\t) {\n"
            for arg in fkt.signature:
                if isinstance(arg, Object): pass
                elif len(arg.shape) > 0:
                    code += "\t\t\t\tif (!(p{0}.incref((PyObject *)PyArray_GETCONTIGUOUS((PyArrayObject *)p{0})))) {{\n".format(arg.name)
                    code += "\t\t\t\t\tPyErr_SetString(PyExc_ValueError, \"Invalid Argument type on {0}!\");\n".format(arg.name)
                    code += "\t\t\t\t\treturn NULL;\n"
                    code += "\t\t\t\t}\n"
                elif arg.dtype is int:
                    if sys.version_info[0] == 2:
                        code += "\t\t\t\tc{0} = PyInt_AS_LONG(p{0});\n".format(arg.name)
                    else:
                        code += "\t\t\t\tc{0} = PyLong_AS_LONG(p{0});\n".format(arg.name)
                elif arg.dtype is float:
                    code += "\t\t\t\tc{0} = PyFloat_AS_DOUBLE(p{0});\n".format(arg.name)
                elif arg.dtype in NPY_SCALAR_TAG:
                    code += "\t\t\t\tc{0} = PyArrayScalar_VAL(p{0}, {1});\n".format(arg.name, NPY_SCALAR_TAG[arg.dtype])

            args = []
            for arg in fkt.signature:
                if not isinstance(arg, Object) and len(arg.shape) > 0:
                    args.append("p{1}, PyArray_SHAPE((PyArrayObject *)p{1}), ({0} *)PyArray_DATA((PyArrayObject *)p{1})".format(PY_C_TYPE[arg.dtype], arg.name))
                else:
                    args.append("c{0}".format(arg.name))

            call  = "{0}_{1}(".format(modtoken.main, fkt.getId())
            call += "\n\t\t\t\t\t\t  {0}".format("\n\t\t\t\t\t\t, ".join(args))
            call += "\n\t\t\t\t\t)"
        else:
            call  = "{0}_{1}()".format(modtoken.main, fkt.getId())

        code += "\t\t\t\ttry {\n"
        if fkt.dtype is None:
            code += "\t\t\t\t\t{0};\n".format(call)
            code += "\t\t\t\t\tPy_INCREF(Py_None);\n"
            code += "\t\t\t\t\treturn Py_None;\n"
        elif len(fkt.shape) == 0 and fkt.dtype is bool:
            code += "\t\t\t\t\tPyObject* res = {0} ? Py_True : Py_False;\n".format(call)
            code += "\t\t\t\t\tPy_INCREF(res);\n"
            code += "\t\t\t\t\treturn res;\n"
        elif len(fkt.shape) == 0 and fkt.dtype is int:
            code += "\t\t\t\t\treturn Py_BuildValue(\"{0}\", {1});\n".format(PY_TYPE_CHAR[np.int_], call)
        elif len(fkt.shape) == 0 and fkt.dtype is float:
            code += "\t\t\t\t\treturn Py_BuildValue(\"{0}\", {1});\n".format(PY_TYPE_CHAR[np.float_], call)
        elif len(fkt.shape) == 0 and fkt.dtype in NPY_SCALAR_TAG:
            code += "\t\t\t\t\tPyObject* res = PyArrayScalar_New({0});\n".format(NPY_SCALAR_TAG[fkt.dtype])
            code += "\t\t\t\t\tPyArrayScalar_ASSIGN(res, {0}, {1});\n".format(NPY_SCALAR_TAG[fkt.dtype], call)
            code += "\t\t\t\t\treturn res;\n"
        else:
            code += "\t\t\t\t\tPyObject * res = std::get<0>({0});\n".format(call)
            if fkt.return_allocated:
                # to avoid mem leak or segfault
                code += "\n\t\t\t\t\tPy_INCREF(res);\n"
                
            code += "\t\t\t\t\treturn res;\n"
        code += "\t\t\t\t} catch (...) {\n"
        code += "\t\t\t\t\treturn NULL;\n"
        code += "\t\t\t\t}\n"

        if len(fkt.signature) > 0:
            code += "\t\t\t} else\n"
            code += "\t\t\t\tPyErr_Clear();\n"

        code += "\t\t}\n"
        
    def stripArg(arg):
        if isinstance(arg, Object):
            delattr(arg, "parent")
            if hasattr(arg, "instance"):
                delattr(arg, "instance")
            for name, value in list(arg.attrs.items()):
                arg.attrs[name] = stripArg(value)
        else:
            if not arg.dtype in [bool, int, float]:
                arg.dtype = NPY_TYPE[arg.dtype]
            arg.dims = len(arg.shape)
            delattr(arg, "shape")
            if not isinstance(arg, ObjectAttr):
                delattr(arg, "scope")
                delattr(arg, "allocated")
            else:
                delattr(arg, "parent")
        return arg

    signatures = []
    for fkt in modtoken.functions[modtoken.main]:
        signatures.append([stripArg(copy.deepcopy(arg)) for arg in fkt.signature])

    if sys.version_info[0] == 2:
        pickled = pickle.dumps(signatures).replace("\n", "\\n")
    else:
        import base64
        pickled = base64.encodebytes(pickle.dumps(signatures)).decode('ascii').replace("\n", "\\n")

    code += "\t\tPyObject * signatures = Py_BuildValue(\"(sO)\", \"{0}\", args);\n".format(pickled)
    code += "\t\tif (!signatures) {\n"

    # TODO: make all exceptions reasonamble: http://docs.python.org/2/c-api/exceptions.html
    code += "\t\t\tPyErr_SetString(PyExc_ValueError, \"Error building signature string for {0}\");\n".format(modtoken.main)
    code += "\t\t\treturn NULL;\n"
    code += "\t\t}\n"
    code += "\t\treturn PyObject_Call(create_signature, signatures, NULL);\n"

    return code
