# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

from __future__ import print_function, division, absolute_import, unicode_literals


import numpy as np

NPY_TYPE = {}
NPY_TYPE[np.int8] = str("int8")
NPY_TYPE[np.int16] = str("int16")
NPY_TYPE[np.int32] = str("int32")
NPY_TYPE[np.int64] = str("int64")
NPY_TYPE[np.uint8] = str("uint8")
NPY_TYPE[np.uint16] = str("uint16")
NPY_TYPE[np.uint32] = str("uint32")
NPY_TYPE[np.uint64] = str("uint64")
NPY_TYPE[np.float32] = str("float32")
NPY_TYPE[np.float64] = str("float64")
NPY_TYPE[int] = NPY_TYPE[np.int_]
NPY_TYPE[float] = NPY_TYPE[np.float_]
if not np.bool_ in NPY_TYPE:
	NPY_TYPE[np.bool_] = str("bool_")
NPY_TYPE[bool] = NPY_TYPE[np.bool_]
if not np.longlong in NPY_TYPE:
	NPY_TYPE[np.longlong] = str("longlong")
if not np.ulonglong in NPY_TYPE:
	NPY_TYPE[np.ulonglong] = str("ulonglong")

PY_C_TYPE = {}
PY_C_TYPE[np.int8] = "npy_int8"
PY_C_TYPE[np.int16] = "npy_int16"
PY_C_TYPE[np.int32] = "npy_int32"
PY_C_TYPE[np.int64] = "npy_int64"
PY_C_TYPE[np.uint8] = "npy_uint8"
PY_C_TYPE[np.uint16] = "npy_uint16"
PY_C_TYPE[np.uint32] = "npy_uint32"
PY_C_TYPE[np.uint64] = "npy_uint64"
PY_C_TYPE[np.float32] = "npy_float"
PY_C_TYPE[np.float64] = "npy_double"
PY_C_TYPE[int] = PY_C_TYPE[np.int_]
PY_C_TYPE[float] = PY_C_TYPE[np.float_]
if not np.bool_ in PY_C_TYPE:
	PY_C_TYPE[np.bool_] = "npy_bool"
PY_C_TYPE[bool] = PY_C_TYPE[np.bool_]
if not np.longlong in PY_C_TYPE:
	PY_C_TYPE[np.longlong] = "npy_longlong"
if not np.ulonglong in PY_C_TYPE:
	PY_C_TYPE[np.ulonglong] = "npy_ulonglong"

PY_TYPE_CHAR = {}
PY_TYPE_CHAR[np.int8] = np.dtype(np.int8).char
PY_TYPE_CHAR[np.int16] = np.dtype(np.int16).char
PY_TYPE_CHAR[np.int32] = np.dtype(np.int32).char
PY_TYPE_CHAR[np.int64] = np.dtype(np.int64).char
PY_TYPE_CHAR[np.uint8] = np.dtype(np.uint8).char
PY_TYPE_CHAR[np.uint16] = np.dtype(np.uint16).char
PY_TYPE_CHAR[np.uint32] = np.dtype(np.uint32).char
PY_TYPE_CHAR[np.uint64] = np.dtype(np.uint64).char
PY_TYPE_CHAR[np.float32] = np.dtype(np.float32).char
PY_TYPE_CHAR[np.float64] = np.dtype(np.float64).char
PY_TYPE_CHAR[int] = "J"
PY_TYPE_CHAR[float] = "D"
if not np.bool_ in PY_TYPE_CHAR:
	PY_TYPE_CHAR[np.bool_] = "p"
PY_TYPE_CHAR[bool] = "P"
if not np.longlong in PY_TYPE_CHAR:
	PY_TYPE_CHAR[np.longlong] = np.dtype(np.longlong).char
if not np.ulonglong in PY_TYPE_CHAR:
	PY_TYPE_CHAR[np.ulonglong] = np.dtype(np.ulonglong).char

NPY_SCALAR_TAG = {}
NPY_SCALAR_TAG[np.int8] = "Int8"
NPY_SCALAR_TAG[np.int16] = "Int16"
NPY_SCALAR_TAG[np.int32] = "Int32"
NPY_SCALAR_TAG[np.int64] = "Int64"
NPY_SCALAR_TAG[np.uint8] = "UInt8"
NPY_SCALAR_TAG[np.uint16] = "UInt16"
NPY_SCALAR_TAG[np.uint32] = "UInt32"
NPY_SCALAR_TAG[np.uint64] = "UInt64"
NPY_SCALAR_TAG[np.float32] = "Float"
NPY_SCALAR_TAG[np.float64] = "Double"
if not np.bool_ in NPY_SCALAR_TAG:
	NPY_SCALAR_TAG[np.bool_] = "Bool"
if not np.longlong in NPY_SCALAR_TAG:
	NPY_SCALAR_TAG[np.longlong] = "LongLong"
if not np.ulonglong in NPY_SCALAR_TAG:
	NPY_SCALAR_TAG[np.ulonglong] = "ULongLong"

NPY_TYPEENUM = {}
NPY_TYPEENUM[np.int8] = "NPY_INT8"
NPY_TYPEENUM[np.int16] = "NPY_INT16"
NPY_TYPEENUM[np.int32] = "NPY_INT32"
NPY_TYPEENUM[np.int64] = "NPY_INT64"
NPY_TYPEENUM[np.uint8] = "NPY_UINT8"
NPY_TYPEENUM[np.uint16] = "NPY_UINT16"
NPY_TYPEENUM[np.uint32] = "NPY_UINT32"
NPY_TYPEENUM[np.uint64] = "NPY_UINT64"
NPY_TYPEENUM[np.float32] = "NPY_FLOAT32"
NPY_TYPEENUM[np.float64] = "NPY_FLOAT64"
NPY_TYPEENUM[int] = NPY_TYPEENUM[np.int_]
NPY_TYPEENUM[float] = NPY_TYPEENUM[np.float_]
if not np.bool_ in NPY_TYPEENUM:
	NPY_TYPEENUM[np.bool_] = "NPY_BOOL"
NPY_TYPEENUM[bool] = NPY_TYPEENUM[np.bool_]
if not np.longlong in NPY_TYPEENUM:
	NPY_TYPEENUM[np.longlong] = "NPY_LONGLONG"
if not np.ulonglong in NPY_TYPEENUM:
	NPY_TYPEENUM[np.ulonglong] = "NPY_ULONGLONG"

UNARY_OPERATORS = {}
UNARY_OPERATORS["UAdd"] = ("+", lambda a: type(+a(1)) )
UNARY_OPERATORS["USub"] = ("-", lambda a: type(-a(1)) )

BINARY_OPERATORS = {}
BINARY_OPERATORS["Add"] = ("+", lambda a, b: type(a(1) + b(1)) )
BINARY_OPERATORS["Sub"] = ("-", lambda a, b: type(a(1) - b(1)) )
BINARY_OPERATORS["Mult"] = ("*", lambda a, b: type(a(1) * b(1)) )
BINARY_OPERATORS["Div"] = ("/", lambda a, b: type(a(1) / b(1)) )
BINARY_OPERATORS["FloorDiv"] = ("//", lambda a, b: type(a(1) // b(1)) )
BINARY_OPERATORS["Pow"] = ("**", lambda a, b: type(a(1) ** b(1)) )
BINARY_OPERATORS["Mod"] = ("%", lambda a, b: type(a(1) % b(1)) )
BINARY_OPERATORS["LShift"] = ("<<", lambda a, b: type(a(1) << b(1)) )
BINARY_OPERATORS["RShift"] = (">>", lambda a, b: type(a(1) >> b(1)) )
BINARY_OPERATORS["BitOr"] = ("|", lambda a, b: type(a(1) | b(1)) )
BINARY_OPERATORS["BitXor"] = ("^", lambda a, b: type(a(1) ^ b(1)) )
BINARY_OPERATORS["BitAnd"] = ("&", lambda a, b: type(a(1) & b(1)) )

BOOL_OPERATORS = {}
BOOL_OPERATORS["And"] = "&&"
BOOL_OPERATORS["Or"] = "||"

AUGMENTED_ASSIGN = {}
AUGMENTED_ASSIGN["Add"] = "+="
AUGMENTED_ASSIGN["Sub"] = "-="
AUGMENTED_ASSIGN["Mult"] = "*="
AUGMENTED_ASSIGN["Div"] = "/="
AUGMENTED_ASSIGN["FloorDiv"] = "//="
AUGMENTED_ASSIGN["Pow"] = "**="
AUGMENTED_ASSIGN["Mod"] = "%="
AUGMENTED_ASSIGN["LShift"] = "<<="
AUGMENTED_ASSIGN["RShift"] = ">>="
AUGMENTED_ASSIGN["BitOr"] = "|="
AUGMENTED_ASSIGN["BitXor"] = "^="
AUGMENTED_ASSIGN["BitAnd"] = "&="

COMPARE_OPERATORS = {}
COMPARE_OPERATORS["Eq"] = ("==", lambda a, b: type(a(1) == b(1)) )
COMPARE_OPERATORS["NotEq"] = ("!=", lambda a, b: type(a(1) != b(1)) )
COMPARE_OPERATORS["Lt"] = ("<", lambda a, b: type(a(1) < b(1)) )
COMPARE_OPERATORS["LtE"] = ("<=", lambda a, b: type(a(1) <= b(1)) )
COMPARE_OPERATORS["Gt"] = (">", lambda a, b: type(a(1) > b(1)) )
COMPARE_OPERATORS["GtE"] = (">=", lambda a, b: type(a(1) >= b(1)) )

# http://docs.scipy.org/doc/numpy/reference/routines.math.html
NPY_UNARY_FUNCTIONS = {}
NPY_UNARY_FUNCTIONS["sin"] = "std::sin"
NPY_UNARY_FUNCTIONS["cos"] = "std::cos"
NPY_UNARY_FUNCTIONS["tan"] = "std::tan"
NPY_UNARY_FUNCTIONS["arcsin"] = "std::asin"
NPY_UNARY_FUNCTIONS["arccos"] = "std::acos"
NPY_UNARY_FUNCTIONS["arctan"] = "std::atan"
# NPY_UNARY_FUNCTIONS["degrees"] = "degrees"
# NPY_UNARY_FUNCTIONS["radians"] = "radians"
# NPY_UNARY_FUNCTIONS["deg2rad"] = "deg2rad"
# NPY_UNARY_FUNCTIONS["rad2deg"] = "rad2deg"
NPY_UNARY_FUNCTIONS["sinh"] = "std::sinh"
NPY_UNARY_FUNCTIONS["cosh"] = "std::cosh"
NPY_UNARY_FUNCTIONS["tanh"] = "std::tanh"
# NPY_UNARY_FUNCTIONS["arcsinh"] = "arcsinh"
# NPY_UNARY_FUNCTIONS["arccosh"] = "arccosh"
# NPY_UNARY_FUNCTIONS["arctanh"] = "arctanh"
NPY_UNARY_FUNCTIONS["floor"] = "std::floor"
NPY_UNARY_FUNCTIONS["ceil"] = "std::ceil"
NPY_UNARY_FUNCTIONS["trunc"] = "std::trunc"
NPY_UNARY_FUNCTIONS["exp"] = "std::exp"
# NPY_UNARY_FUNCTIONS["expm1"] = "expm1"
# NPY_UNARY_FUNCTIONS["exp2"] = "exp2"
NPY_UNARY_FUNCTIONS["log"] = "std::log"
# NPY_UNARY_FUNCTIONS["log10"] = "log10"
# NPY_UNARY_FUNCTIONS["log2"] = "log2"
# NPY_UNARY_FUNCTIONS["log1p"] = "log1p"
# NPY_UNARY_FUNCTIONS["iO"] = "iO"
# NPY_UNARY_FUNCTIONS["sinc"] = "sinc"
# NPY_UNARY_FUNCTIONS["signbit"] = "signbit"
# NPY_UNARY_FUNCTIONS["frexp"] = "frexp"
NPY_UNARY_FUNCTIONS["sqrt"] = "std::sqrt"
# NPY_UNARY_FUNCTIONS["square"] = "square"
NPY_UNARY_FUNCTIONS["abs"] = "abs"
# NPY_UNARY_FUNCTIONS["absolute"] = "absolute"
NPY_UNARY_FUNCTIONS["fabs"] = "std::fabs"
NPY_UNARY_FUNCTIONS["sign"] = "sign"

NPY_BINARY_FUNCTIONS = {}
# NPY_BINARY_FUNCTIONS["hypot"] = "hypot"
# NPY_BINARY_FUNCTIONS["arctan2"] = "arctan2"
# NPY_BINARY_FUNCTIONS["logaddexp"] = "logaddexp"
# NPY_BINARY_FUNCTIONS["logaddexp2"] = "logaddexp2"
# NPY_BINARY_FUNCTIONS["copysign"] = "copysign"
# NPY_BINARY_FUNCTIONS["ldexp"] = "ldexp"
# NPY_BINARY_FUNCTIONS["power"] = "power"
# NPY_BINARY_FUNCTIONS["convolve"] = "convolve"
# NPY_BINARY_FUNCTIONS["maximum"] = "maximum"
# NPY_BINARY_FUNCTIONS["minimum"] = "minimum"
# NPY_BINARY_FUNCTIONS["fmax"] = "fmax"
# NPY_BINARY_FUNCTIONS["fmin"] = "fmin"

# Numpy type conversions
NPY_CAST_FUNCTIONS = {}
NPY_CAST_FUNCTIONS["bool_"] = np.bool_
NPY_CAST_FUNCTIONS["int_"] = np.int_
NPY_CAST_FUNCTIONS["intc"] = np.intc
NPY_CAST_FUNCTIONS["int8"] = np.int8
NPY_CAST_FUNCTIONS["int16"] = np.int16
NPY_CAST_FUNCTIONS["int32"] = np.int32
NPY_CAST_FUNCTIONS["int64"] = np.int64
NPY_CAST_FUNCTIONS["uint8"] = np.uint8
NPY_CAST_FUNCTIONS["uint16"] = np.uint16
NPY_CAST_FUNCTIONS["uint32"] = np.uint32
NPY_CAST_FUNCTIONS["uint64"] = np.uint64
NPY_CAST_FUNCTIONS["float_"] = np.float_
NPY_CAST_FUNCTIONS["float32"] = np.float32
NPY_CAST_FUNCTIONS["float64"] = np.float64
