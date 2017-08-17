# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate the supported NumPy functions

http://pythonhosted.org/hope/lang.html#numpy-mathematical-functions

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import hope

prefix = "hope_numpy"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

@hope.jit
def numpy_math(x):
    # **Trigonometric functions**
    a = np.sin(x)     #   Trigonometric sine, element-wise.
    a = np.cos(x)     #   Cosine elementwise.
    a = np.tan(x)     #   Compute tangent element-wise.
    a = np.arcsin(x)     #Inverse sine, element-wise.
    a = np.arccos(x)     #Trigonometric inverse cosine, element-wise.
    a = np.arctan(x)     #Trigonometric inverse tangent, element-wise.
    
    # **Hyperbolic functions**
    a = np.sinh(x)     #  Hyperbolic sine, element-wise.
    a = np.cosh(x)     #  Hyperbolic cosine, element-wise.
    a = np.tanh(x)     #  Compute hyperbolic tangent element-wise.
    
    
    # **Miscellaneous**
    a = np.exp(x)     #   Calculate the exponential of all elements in the input array.
    a = np.sum(x)     #          Return the sum of array elements.
    a = np.sqrt(x)     #         Return the positive square-root of an array, element-wise.
    a = np.ceil(x)     #         Return the ceiling of the input, element-wise.
    a = np.floor(x)     #        Return the floor of the input, element-wise.
    a = np.trunc(x)     #        Return the truncated value of the input, element-wise.
    a = np.fabs(x)     #            Compute the absolute values element-wise
    a = np.pi     #              Returns the pi constant

@hope.jit
def numpy_interp(x, xp, fp):
    return np.interp(x, xp, fp)

def example():
    numpy_math(1.5)
    
    xp = np.array([0.0, 1.0])
    fp = np.array([1.0, 2.0])
    x = 0.5
    r = numpy_interp(x, xp, fp)
    assert r == 1.5

    x = np.array([0.25, 0.5, 0.75])
    r = numpy_interp(x, xp, fp)
    assert np.all(r == [1.25, 1.5, 1.75])
    
if __name__ == '__main__':
    example()
