# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate if-elif-else support

http://pythonhosted.org/hope/lang.html#conditional-expressions

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil

import hope

prefix = "hope_conditions"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

@hope.jit
def demo(d):
    if d == 0:
        return d + 1
    elif d == 1:
        return d
    else:
        return d -1 

def example():
    r = demo(0)
    assert r == 1

    r = demo(1)
    assert r == 1

    r = demo(2)
    assert r == 1

if __name__ == '__main__':
    example()