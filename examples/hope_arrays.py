# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate the supported functionality with NumPy arrays

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import hope

prefix = "hope_arrays"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

@hope.jit
def create_empty(n):
    return np.empty(n)

@hope.jit
def array_slicing_1D():
    a = np.zeros(4)
    a[:] = 3 
    a[:2] = 1
    a[1:2] = 2
    a[3:] = 4
    return a

@hope.jit
def array_slicing_2D(a):
    a[:] = 3
    a[1:, :] = 1
    a[:, 1:] = 2
    a[2:, 2:] = 4
    a[3, 3] = 5
    a[1, :-2] = 6
    a[:-2, 1] = 7
    a[0, -1] = 8
    return a

@hope.jit
def array_asignment(a, b):
    a[2:] = b[:2]

def example():
    #create new arrays (same for ones and zeros). Only 1D
    r = create_empty(5)
    assert len(r) == 5
    
    #slicing 1D array to assign new values
    r = array_slicing_1D()
    print(r)
    assert sum(r) == 10
    
    #slicing 2D array to assign new values
    a = np.zeros((4,4))
    r = array_slicing_2D(a)
    print(r)
    assert np.sum(r) == 60
    
    #assign a slice of an array to a slice
    a = np.zeros(4)
    b = np.ones(4)
    array_asignment(a, b)
    print(a)
    assert np.sum(a) == 2
    
if __name__ == '__main__':
    example()