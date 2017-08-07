# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
Created on Aug 26, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import pytest
import ast
import itertools
import hope
from hope import _tosource
from hope import _transformer


def dummy(par, *args, **kwargs):
    for _ in range(1):
        continue
    assert True, "msg"


def dummy2():
    """used to test for differences between ast of python 3.4 and previous versions"""
    kws = dict(c=7)
    args = (1, 2)
    return dummy(1, *args, **kws) + dummy(2, **{'c': 7})


@pytest.mark.infrastructure
class TestToSource(object):

    def test_tosource_mod(self):
        self.exec_test_on_mod(hope._tosource)

    def test_wrapper_mod(self):
        self.exec_test_on_mod(hope._wrapper)

    def test_simple(self):
        mod_ast = _transformer.get_fkt_ast(dummy)
        mod_source = _tosource.tosource(mod_ast)
#         with open("dummy.py", "w") as f:
#             f.write(mod_source)
        mod_ast2 = ast.parse(mod_source)
        assert compare_ast(mod_ast, mod_ast2.body[0])

    def test_simple_2(self):
        mod_ast = _transformer.get_fkt_ast(dummy2)
        mod_source = _tosource.tosource(mod_ast)
        mod_ast2 = ast.parse(mod_source)
        assert compare_ast(mod_ast, mod_ast2.body[0])

    def exec_test_on_mod(self, mod):
        modpath = mod.__file__
        if modpath.endswith('.pyc'):
            modpath = modpath[:-1]

        with open(modpath, "r") as f:
            source = f.read()
        mod_ast = ast.parse(source)
        mod_source = _tosource.tosource(mod_ast)
        mod_source = mod_source.replace("'\\n'", "@@@@")
        mod_source = mod_source.replace("\\n", "\n")
        mod_source = mod_source.replace("@@@@", "'\\n'")

#         with open("dummy.py", "w") as f:
#             f.write(mod_source)
        mod_ast2 = ast.parse(mod_source)
        assert compare_ast(mod_ast, mod_ast2)

def compare_ast(node1, node2):
    if type(node1) is not type(node2):
        raise Exception(node1, node2)
        return False
    if isinstance(node1, ast.AST):
        for k, v in vars(node1).items():
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast, zip(node1, node2)))
    else:
        return node1 == node2
