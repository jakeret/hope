# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Created on Mar 11, 2014

author: jakeret
"""

from __future__ import print_function, division, absolute_import, unicode_literals

from operator import eq
from mock import patch
from mock import MagicMock

from hope.options import get_cxxflags
from hope import options
import pytest
from hope.exceptions import UnsupportedCompilerException

DARWIN_COMPILERS = ["gcc-mac", "clang", "gcc", "cc", "c++", "icc"]
DARWIN_MAPPING = {"gcc": "gcc-mac",
                  "cc": "clang",
                  "c++": "clang",
                  }

LINUX_COMPILERS = ["gcc-linux", "gcc", "cc", "c++", "icc", "x86_64-linux-gnu-gcc"]
LINUX_MAPPING = {"gcc": "gcc-linux",
                  "cc": "gcc",
                  "c++": "gcc",
                  "x86_64-linux-gnu-gcc": "gcc-linux"
                  }

# custom marker enabling selective unit testing
# e.g. py.test -v -m "infrastructure"

@pytest.mark.infrastructure
class TestOptions(object):


    @pytest.mark.parametrize("compiler_name", DARWIN_COMPILERS)
    def test_get_cxxflags_Darwin(self, compiler_name):
        with patch("platform.system") as sys_mock:
            sys_mock.return_value = options.DARWIN_KEY
            with patch("distutils.ccompiler.new_compiler") as new_cc_mock:
                with patch("subprocess.check_output") as command_mock:
                    command_mock.return_value = options.MIN_GCC_VERSION.encode('utf-8')
                    cc_mock = MagicMock()
                    cc_mock.compiler = [compiler_name]
                    new_cc_mock.return_value = cc_mock
                    flags = get_cxxflags()
                    assert flags is not None
                    
                    try:
                        compiler = DARWIN_MAPPING[compiler_name]
                    except KeyError:
                        compiler = compiler_name
                    
                    assert map(eq, flags, options.CXX_FLAGS[compiler])

    @pytest.mark.parametrize("compiler_name", LINUX_COMPILERS)
    def test_get_cxxflags_Linux(self, compiler_name):
        with patch("platform.system") as sys_mock:
            sys_mock.return_value = options.LINUX_KEY
            with patch("distutils.ccompiler.new_compiler") as new_cc_mock:
                with patch("subprocess.check_output") as command_mock:
                    command_mock.return_value = options.MIN_GCC_VERSION.encode('utf-8')
                    cc_mock = MagicMock()
                    cc_mock.compiler = [compiler_name]
                    new_cc_mock.return_value = cc_mock
                    flags = get_cxxflags()
                    assert flags is not None
                    
                    try:
                        compiler = LINUX_MAPPING[compiler_name]
                    except KeyError:
                        compiler = compiler_name
                    
                    assert map(eq, flags, options.CXX_FLAGS[compiler])
                
    @pytest.mark.parametrize("compiler_name", ["gcc-linux", "gcc"]) 
    def test_get_cxxflags_unsupported_version(self, compiler_name):
        with patch("platform.system") as sys_mock:
            sys_mock.return_value = options.LINUX_KEY
            with patch("distutils.ccompiler.new_compiler") as new_cc_mock:
                with patch("subprocess.check_output") as command_mock:
                    command_mock.return_value = "1.0.0".encode('utf-8')
                    cc_mock = MagicMock()
                    cc_mock.compiler = [compiler_name]
                    new_cc_mock.return_value = cc_mock
                    
                    try:
                        get_cxxflags()
                        pytest.fail("Version should not be supported")
                    except UnsupportedCompilerException:
                        assert True
    
    def test_get_cxxflags_invalid_os(self):
        with patch("platform.system") as sys_mock:
            sys_mock.return_value = "Win32"
            
            try:
                flags = get_cxxflags()
                pytest.fail("OS is not supported")
            except UnsupportedCompilerException:
                assert True

    def test_get_cxxflags_invalid_compiler(self):
        with patch("platform.system") as sys_mock:
            sys_mock.return_value = options.LINUX_KEY
            with patch("distutils.ccompiler.new_compiler") as new_cc_mock:
                cc_mock = MagicMock()
                cc_mock.compiler = ["invalid_cc"]
                new_cc_mock.return_value = cc_mock
                try:
                    flags = get_cxxflags()
                    pytest.fail("Compiler is not supported")
                except UnsupportedCompilerException:
                    assert True

    def test_get_cxxflags_gcc_clang(self):
        compiler_name = "gcc"
        with patch("platform.system") as sys_mock:
            sys_mock.return_value = options.DARWIN_KEY
            with patch("distutils.ccompiler.new_compiler") as new_cc_mock:
                with patch("subprocess.check_output") as command_mock:
                    command_mock.return_value = options.GCC_CLANG_VERSION.encode('utf-8')
                    cc_mock = MagicMock()
                    cc_mock.compiler = [compiler_name]
                    new_cc_mock.return_value = cc_mock
                    flags = get_cxxflags()
                    assert flags is not None
                    
                    try:
                        compiler = DARWIN_MAPPING[compiler_name]
                    except KeyError:
                        compiler = compiler_name
                    
                    assert map(eq, flags, options.CXX_FLAGS[compiler])

if __name__ == '__main__':
    test = TestOptions()
    test.test_get_cxxflags_Darwin()
