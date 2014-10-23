# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate loop support

http://pythonhosted.org/hope/lang.html#loops

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil

import hope

prefix = "hope_loops"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

@hope.jit
def while_loop(d):
    while d > 0:
        d = d -1
    return d 

@hope.jit
def range_end(d):
    s = 0   
    for i in range(d):
        s = s + i
    return s

@hope.jit
def range_start_end(d):
    s = 0   
    for i in range(d, d*2):
        s = s + i
    return s


def example():
    r = while_loop(2)
    assert r == 0

    #equivalent support for xrange
    r = range_end(5)
    assert r == 10
    r = range_start_end(5)
    assert r == 35

if __name__ == '__main__':
    example()