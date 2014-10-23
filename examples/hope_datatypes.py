# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate some of the supported data types

http://pythonhosted.org/hope/lang.html#native-types
http://pythonhosted.org/hope/lang.html#numpy-types

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np

import hope

prefix = "hope_datatypes"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

@hope.jit
def demo(d):
    return d

def example():
    #built-in
    demo(1)
    demo(1.0)
     
    #numpy 1d array
    demo(np.array([1], dtype=np.int8))
    demo(np.array([1], dtype=np.uint8))
    demo(np.array([1], dtype=np.float32))
    
    #numpy 2d array
    demo(np.array([[1], [2]], dtype=np.int8))
    demo(np.array([[1], [2]], dtype=np.uint16))
    demo(np.array([[1], [2]], dtype=np.float32))
    
    
    
if __name__ == '__main__':
    example()