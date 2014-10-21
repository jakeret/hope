# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
Created on Sep 26, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import pytest
from hope import _wrapper
import hope
from hope.jit import _check_state, jit

def dummy():
    pass

def dummy_called():
    pass

@pytest.mark.infrastructure
class TestJit(object):
    
    @pytest.fixture
    def config_state(self):
        state = {}
        for name in _wrapper.get_config_attrs():
            state[name] = getattr(hope.config, name)
            
        state["filename"] = "filename"
        return state
    
    def test_inconsistent_config(self):
        state = {"cxxflags": "new_flag"}
        
        try:
            _check_state(None, state)
            pytest.fail("Config is inconsistent")
        except LookupError as le:
            assert True
            
    def test_no_fkt_in_state(self, config_state):
        try:
            _check_state(None, config_state)
            pytest.fail("Config is inconsistent")
        except LookupError as le:
            assert True
            
        config_state["main"] = "fkt_name"
        try:
            _check_state(None, config_state)
            pytest.fail("Config is inconsistent")
        except LookupError as le:
            assert True
            
        config_state["called"] = "called_fkt_name"
        try:
            _check_state(dummy, config_state)
            pytest.fail("Config is inconsistent")
        except LookupError as le:
            assert True
            
    def test_missing_calling_fkt(self, config_state):
        config_state["main"] = dummy.__name__
        config_state["called"] = {}
        _check_state(dummy, config_state)
        
        config_state["called"]["called1"] = "hash"
        try:
            _check_state(dummy, config_state)
            pytest.fail("Config is inconsistent")
        except LookupError as le:
            assert True

        config_state["called"].pop("called1")
        config_state["called"][dummy_called.__name__] = "hash"
        try:
            _check_state(dummy, config_state)
            pytest.fail("Config is inconsistent")
        except LookupError as le:
            assert True

        config_state["called"].pop(dummy_called.__name__)
        config_state["called"][_check_state.__name__] = "hash"
        try:
            _check_state(dummy, config_state)
            pytest.fail("Config is inconsistent")
        except LookupError as le:
            assert True

    def test_fkt_with_args(self):
        with pytest.raises(ValueError):
            @jit
            def func_with_star_args(a, *args):
                pass
        with pytest.raises(ValueError):
            @jit
            def func_with_kwargs(a, **kwargs):
                pass
        with pytest.raises(ValueError):
            @jit
            def func_with_star_args_and_kwargs(a, *args, **kwargs):
                pass
        # Should not raise ValueError
        @jit
        def func_with_vanilla_args(a, b):
            pass
