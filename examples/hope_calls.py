# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate the support for calling other functions

http://pythonhosted.org/hope/lang.html#call-functions

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil

import hope

prefix = "hope_calls"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

def fkt1(d):
    return d + 1 

@hope.jit
def demo1(d):
    s = fkt1(d)
    return s

@hope.jit
def fkt2(d):
    return d + 1 

@hope.jit
def demo2(d):
    s = fkt2(d)
    return s

@hope.jit
def fkt_recursion(n):
    if n < 2:
        return n
    return fkt_recursion(n - 1) + fkt_recursion(n - 2)

def example():
    #jitted function calls non-decorated function
    r = demo1(2)
    assert r == 3
    
    fkt2(0)
    #jitted function calls decorated function
    r = demo2(2)
    assert r == 3
    
    #recursive calls of a jitted function
    assert fkt_recursion(20) == 6765

if __name__ == '__main__':
    example()