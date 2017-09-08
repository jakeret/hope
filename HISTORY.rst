.. :changelog:

History
-------

0.7.0 (2017-09-08)
++++++++++++++++++

* Support for Python 3.5 and 3.6


0.6.1 (2016-07-04)
++++++++++++++++++

* fixing bug when accessing class members for operations

0.6.0 (2016-04-19)
++++++++++++++++++

* Fixed bug in 2d array slicing
* Array slicing with negative index
* Fixed name clash bug with object attributes
* Replaced assignment with reference to object attributes 

0.5.0 (2016-01-20)
++++++++++++++++++

* Fixed memory leak when creating array in jitted fkt
* Fixed incorrect bound handling in numpy.interp

0.4.0 (2015-02-04)
++++++++++++++++++

* Increased compilation speed for large functions
* Support for variable allocation within if-else
* Added support for numpy.sign
* Updated Cython implementation in benchmarks
* Fixed array allocation problem under OSX Yosemite (thx iankronquist)

0.3.1 (2014-10-24)
++++++++++++++++++

* Better support for Python 3.3 and 3.4
* Proper integration in Travis CI
* Improved support for Linux systems (`accepting x86_64-linux-gnu-gcc`)
* Avoiding warning on Linux by removing `Wstrict-prototypes` arg
* Supporting gcc proxied clang (OS X)
* Added set of examples

0.3.0 (2014-10-16)
++++++++++++++++++

* Language: scalar return values
* Shared libraries are written to hope.config.prefix
* Function call can have return values
* Fixed function calls to function with no arguments
* Make sure code is recompiled if the python code has changed
* Added config.optimize to simplify expression using sympy and replace pow
* Speed improvements for hope
* Added support for object properties
* Added support for object methods
* Addes support for True and False
* Addes support for While
* Addes support for numpy.sum
* Addes support for numpy.pi
* Added support for numpy.floor, numpy.ceil, numpy.trunc, numpy.fabs, numpy.log
* improved error messages
* Added config.rangecheck flag
* Support xrange in for loop
* Added cast operators for np.bool\_, np.int\_, np.intc, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float\_, np.float32, np.float64, 
* Added bool operators
* Added the following operators:

===========   ===========
FloorDiv      ``a // b``
Mod           ``a % b``
LShift        ``a << b``
RShift        ``a >> b``
BitOr         ``a | b``
BitXor        ``a ^ b``
BitAnd        ``a & b``
AugFloorDiv   ``a //= b``
AugPow        ``a **= b``
AugMod        ``a %= b``
AugLShift     ``a <<= b``
AugRShift     ``a <<= b``
AugBitOr      ``a | b``
AugBitXor     ``a ^ b``
AugBitAnd     ``a & b``
===========   ===========

0.2.0 (2014-03-05)
++++++++++++++++++

* First release on private PyPI.

0.1.0 (2014-02-27)
++++++++++++++++++

* Initial creation.
