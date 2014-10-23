# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate the supported operators

http://pythonhosted.org/hope/lang.html#operators

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import hope

prefix = "hope_operators"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

@hope.jit
def assign(d):
    a = d
    return a

@hope.jit
def invert(d):
    a = -d
    return a

@hope.jit
def add(a, b):
    return a + b

def example():
    #assigning
    d = 1 
    r = assign(d)
    assert r == d

    d = np.array([1])
    r = assign(d)
    assert np.all(r == d)
    
    d = np.array([[1], [2]])
    r = assign(d)
    assert np.all(r == d)

    #unary operation
    d = 1 
    r = invert(d)
    assert r == -d

    d = np.array([1])
    r = invert(d)
    assert np.all(r == -d)
    
    d = np.array([[1], [2]])
    r = invert(d)
    assert np.all(r == -d)
    
    #add (same for -,*, /)
    a, b = 1, 1 
    r = add(a, b)
    assert r == a + b

    a, b = np.array([1]), np.array([1])
    r = add(a, b)
    assert np.all(r == a + b)
    
    a, b = np.array([[1], [2]]), np.array([[1], [2]])
    r = add(a, b)
    assert np.all(r == a + b)

if __name__ == '__main__':
    example()