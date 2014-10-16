Optimization
============

After the **HOPE** specific AST has been created the package performs a static recursive analysis of the expressions to introduce numerical optimization. The supported possibilities are divided into three groups: 

Simplification of expressions
----------------------------- 

To simplify expression we have used the ``SymPy`` `library <http://sympy.org>`_. SymPy is a Python library for symbolic mathematics and has been entirely written in Python. To apply the optimization, the AST expression is translated into ``SymPy`` syntax AST and passed to the ``simplify`` function. The function applies various different heuristics to reduce the complexity of the passed expression. The simplification is not exactly defined and varies depending on the input. 

For instance, one example of simplification is that :math:`sin(x)^2 + cos(x)^2` will be simplified to :math:`1`.

Factorizing out subexpressions
------------------------------ 

Furthermore the ``SymPy`` library is used to factorize out recurring subexpression (common subexpression elimination) using the previously created ``SymPy`` AST and ``SymPy``'s ``cse`` function.

Replacing the pow function for integer exponents
---------------------------------------------------- 

From C++11 on, the ``pow`` function in the C standard library is not `overloaded for integer exponents <http://en.cppreference.com/w/cpp/numeric/math/pow>`_. The internal implementation of the computation of a base to the power of a double exponent is typically done using a series expansion, though this may vary depending on the compiler and hardware architecture. Generally this is efficient for double exponents but not necessarily for integer exponents. 

**HOPE** therefore tries to identify power expressions with integer exponents and factorizes the expression into several multiplications e.g. :math:`y=x^5` will be decomposed into :math:`x_2=x^2` and :math:`y=x_2\times x_2 \times x`. This reduces the computational costs and increases the performance of the execution.
