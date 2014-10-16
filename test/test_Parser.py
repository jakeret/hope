# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
Created on Aug 22, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import pytest
from hope._transformer import get_fkt_ast
import ast

def dummy(a, b):
    return a + b

@pytest.mark.infrastructure
class TestParser(object):
    
    def test_get_fkt_ast_simple(self):
        fkt_ast = get_fkt_ast(dummy)
        assert fkt_ast is not None
        assert isinstance(fkt_ast, ast.FunctionDef)
        assert len(fkt_ast.body)==1
        assert isinstance(fkt_ast.body[0], ast.Return)
        
    #TODO: how can I cause an exception? Add a new test
