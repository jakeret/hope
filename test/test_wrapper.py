# Copyright (C) 2013 ETH Zurich, Institute for Astronomy

'''
Created on Aug 8, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from hope import _wrapper
import os
import pytest
from py._path.local import LocalPath
from hope._ast import FunctionDef

def dummy_fkt():
    return 0

@pytest.mark.infrastructure
class TestWrapper(object):
    
    cpp_hello_world = """
                        int main()
                        {
                            return 0;
                        }"""

    def test_compile(self, tmpdir):
        tmp_path = str(tmpdir)
        localfilename = "helloworld"
        with open(os.path.join(tmp_path, localfilename+".cpp"), "w") as source:
            source.write(self.cpp_hello_world)
        
        _wrapper._compile(tmp_path, localfilename, localfilename)
        
        assert os.path.exists(os.path.join(tmp_path, localfilename+".out"))
        
        
    def test_compile_invalid(self, tmpdir):
        tmp_path = str(tmpdir)
        localfilename = "invalid"
        with open(os.path.join(tmp_path, localfilename+".cpp"), "w") as source:
            source.write("invalid")
        
        try:
            _wrapper._compile(tmp_path, localfilename, localfilename)
            pytest.fail("Source can't be compiled!")
        except Exception:
            assert True        


    def test_store_state(self, tmpdir):
        tmp_path = str(tmpdir)
        localfilename = "dummy_fkt"
        
        wrapper = _wrapper.Wrapper(dummy_fkt, "0")
        wrapper.modtoken.functions[localfilename] = [FunctionDef(localfilename, [])]
        wrapper._store_state(tmp_path, localfilename)
        
        assert os.path.exists(os.path.join(tmp_path, "{0}.pck".format(wrapper.filename)))
        
