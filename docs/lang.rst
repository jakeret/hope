Language Specification
======================

This document specifies the Python subset supported by **HOPE**.

Native Types
------------
The following native types are supported

- ``bool``
- ``int``
- ``float``

NumPy Types
-----------
The following NumPy types are supported

- ``bool_``
- ``integer``, ``signedinteger``, ``byte``, ``short``, ``intc``, ``intp``, ``int0``, ``int_``, ``longlong``
- ``int8``, ``int16``, ``int32``, ``int64``
- ``unsignedinteger``, ``ubyte``, ``ushort``, ``uintc``, ``uintp``, ``uint0``, ``uint_``, ``ulonglong``
- ``uint8``, ``uint16``, ``uint32``, ``uint64``
- ``single``, ``float_``
- ``float32``, ``float64``
- ``ndarray``

Conditional Expressions
-----------------------
- If
- If/Else
- If/ElseIf/Else

Loops
-----
The ``while`` statement is supported as well as ``for`` loops but only with ``range(stop)`` or ``range(start, stop)`` resp. ``xrange``::

    for i in range(start, stop):
        foo()

Return Statement
----------------
A function needs to have a fixed return type. **HOPE** currently supports scalar and array data types as return values. 

The following code will not compile as the type of the return value may change depending on the execution: 

::

    @hope.jit
    def incompatible_return(arg):
        if arg > 10:
            return 1
        else:
            return 2.3 # ERROR: Inconsistent return type

Call functions
--------------
Call to pure Python functions are supported if the function

- is accessible form the global scope of the function
- has no decorators
- only uses the subset of Python supported by **HOPE** 
- has no recursive or cyclic calls

Then the called function is also compiled to c++ and included in the shared object
regardless where the function was defined originally.

Operators
---------

**Assignment**

========== =========
Assign     ``b = a``
========== =========

**Unary operators**

========== =========
UAdd       ``+a``
USub       ``-a``
========== =========

**Binary operators**

========== =========
Add        ``a + b``
Sub        ``a - b``
Mult       ``a * b``
Div        ``a / b``
FloorDiv   ``a // b``
Pow        ``a ** b``
Mod        ``a % b``
LShift     ``a << b``
RShift     ``a >> b``
BitOr      ``a | b``
BitXor     ``a ^ b``
BitAnd     ``a & b``
========== =========

**Augmented assign statements**

=========== ===========
AugAdd      ``a += b``
AugSub      ``a -= b``
AugMult     ``a *= b``
AugDiv      ``a /= b``
AugFloorDiv ``a /= b``
AugPow      ``a **= b``
AugMod      ``a %= b``
AugLShift   ``a <<= b``
AugRShift   ``a <<= b``
AugBitOr    ``a | b``
AugBitXor   ``a ^ b``
AugBitAnd   ``a & b``
=========== ===========

**Comparison Operators**

=========== =========
Eq          ``a == b``
NotEq       ``a != b``
Lt          ``a < b``
LtE         ``a <= b``
Gt          ``a > b``
GtE         ``a >= b``
=========== =========

**Bool Operators**

==== ============
&&   ``a and b``
||   ``a or b``
==== ============


NumPy Array creation routines
-----------------------------

============================= =======================================================================================
``empty(shape[, dtype])``     Return a new array of given shape and type, without initializing entries.
``ones(shape[, dtype])``      Return a new array of given shape and type, filled with ones.
``zeros(shape[, dtype])``     Return a new array of given shape and type, filled with zeros.
============================= =======================================================================================

NumPy Mathematical functions
----------------------------

**Trigonometric functions**

============================= =======================================================================================
``sin(x)``                    Trigonometric sine, element-wise.
``cos(x)``                    Cosine elementwise.
``tan(x)``                    Compute tangent element-wise.
``arcsin(x)``                 Inverse sine, element-wise.
``arccos(x)``                 Trigonometric inverse cosine, element-wise.
``arctan(x)``                 Trigonometric inverse tangent, element-wise.
============================= =======================================================================================

**Hyperbolic functions**

============================= =======================================================================================
``sinh(x)``                   Hyperbolic sine, element-wise.
``cosh(x)``                   Hyperbolic cosine, element-wise.
``tanh(x)``                   Compute hyperbolic tangent element-wise.
============================= =======================================================================================

**Exponents and logarithms**

============================= =======================================================================================
``exp(x)``                    Calculate the exponential of all elements in the input array.
============================= =======================================================================================

**Miscellaneous**

==================================== =======================================================================================
``sum(x)``                           Return the sum of array elements.
``sqrt(x)``                          Return the positive square-root of an array, element-wise.
``interp(x, xp, fp[, left, right])`` One-dimensional linear interpolation.
``ceil(x)``                          Return the ceiling of the input, element-wise.
``floor(x)``                         Return the floor of the input, element-wise.
``trunc(x)``                         Return the truncated value of the input, element-wise.
``pi``                               Returns the pi constant
``fabs``                             Compute the absolute values element-wise
``sign``                             Returns an element-wise indication of the sign of a number
==================================== =======================================================================================


Attributes of ``numpy.ndarray``
-------------------------------
No attributes are supported at the moment

Others
------

* Added cast operators for np.bool\_, np.int\_, np.intc, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float\_, np.
