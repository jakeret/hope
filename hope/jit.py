# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

from __future__ import print_function, division, absolute_import, unicode_literals


import os
import sys
import hashlib
import inspect
import warnings

from hope._wrapper import Wrapper
import hope._cache as cache
from hope import config
from hope import serialization
from hope._wrapper import get_config_attrs
from hope._wrapper import get_fkt_hash

# TODO: add test for hope.pow
# TODO: add test and doc for additional numpy functions
# TODO: implement self, self.cost, self.fkt
# TODO: replace function in global namespace by c pointer to get the native speed on the first run
# TODO: make two versions for np.int_ and int
# TODO: make hope.interp with retuns a callable object with 2^n basepoints and c pendant
# TODO: if dtype of argument is alreadu correct, do not cast it in c code
# TODO: add test for subscrpto with all types to get the same semantics
# TODO: controll structures
# TODO: check for PyArray_ISNOTSWAPPED
# TODO: merge scalar blocks and search subexpressions in all lines
# TODO: asserts of sizes
# TODO: make class derived from tuple which can hold several parameters and be passed to hope functions
# TODO: use int PyArrayObject.flags to check if data has to be copied http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#PyArrayObject
# TODO: use https://github.com/workhorsy/py-cpuinfo/blob/master/cpuinfo.py to detect features
# TODO: other code generator: http://documen.tician.de/codepy/jit.html#module-codepy.jit
# TODO: make ufunc decorator http://docs.scipy.org/doc/numpy/reference/ufuncs.html like numba http://numba.pydata.org/numba-doc/0.12.1/ufuncs.html
# TODO: optimize pow with interger powers to multipications
# TODO: make tests for _dump
# TODO: add constants/class constants to hope
# TODO: use sympy to simpify and Common Subexpression Detection
# 		http://docs.sympy.org/latest/modules/rewriting.html
# 		http://docs.sympy.org/latest/modules/core.html#module-sympy.core.sympify


def jit(fkt):
    """
    Compiles a function to native code and return the optimized function. The new function has the performance of a compiled function written in C.

    :param fkt: function to compile to c
    :type fkt: function
    :returns: function -- optimized function

    This function can either be used as decorator

    .. code-block:: python

        @jit
        def sum(x, y):
            return x + y

    or as a normal function

    .. code-block:: python

        def sum(x, y):
            return x + y
        sum_opt = jit(sum)
    """

    if config.hopeless:
        return fkt

    argspec = inspect.getargspec(fkt)
    if argspec.varargs is not None or argspec.keywords is not None:
        raise ValueError("Jitted functions should not have *args or **kwargs")

    hash = hashlib.sha224(inspect.getsource(fkt).encode('utf-8')).hexdigest()
    filename = "{0}_{1}".format(fkt.__name__, hash)

    if not os.path.exists(config.prefix):
        os.makedirs(config.prefix)

    if not config.prefix in sys.path:
        sys.path.append(os.path.abspath(config.prefix))

    wrapper = Wrapper(fkt, hash)

    try:
        state = serialization.unserialize(filename)
        if not state is None:
            _check_state(fkt, state)
        else:
            raise ImportError("No state found.")

        if sys.version_info[0] == 2:
            module = __import__(state["filename"], globals(), locals(), [], -1)
        else:
            import importlib
            module = importlib.import_module(state["filename"])

        if "bound" in state and state["bound"]:
            def _hope_callback(*args):
                return module.run(*args)
            setattr(cache, str(id(_hope_callback)), fkt)
            return _hope_callback
        else:
            module.set_create_signature(wrapper.create_signature)
            setattr(cache, str(id(module.run)), fkt)
            return module.run

    except LookupError as le:
        if config.verbose:
            warnings.warn("Recompiling... Reason: {0}".format(le))
        wrapper._recompile = True
        return wrapper.callback
    except ImportError as ie:
        return wrapper.callback

def _check_state(fkt, state):
    for name in get_config_attrs():
        if name not in state or state[name] != getattr(config, name):
            raise LookupError("State is inconsistent with config. Inconsistent state key: [{0}].".format(name))
        
    if "main" not in state or "called" not in state or state["main"] != fkt.__name__:
        raise LookupError("State is inconsistent")
    
    for name, value in list((state["called"] if "called" in state else {}).items()):
        if name not in fkt.__globals__:
            #TODO: FIX! state of globals depends on the order of function in module. If called function comes later in the code we raise the error
            raise LookupError("State is inconsistent. Called function '%s' cannot be found in %s's global scope."%(name, fkt.__name__))

        glob_fkt = fkt.__globals__[name]
        if isinstance(glob_fkt, Wrapper):
            if "filename" in state and get_fkt_hash(glob_fkt.fkt) != value:
                raise LookupError("State is inconsistent. Hash(sha224) has changed")
        elif inspect.isbuiltin(glob_fkt) and hasattr(cache, str(id(glob_fkt))):
            if "filename" in state and get_fkt_hash(getattr(cache, str(id(glob_fkt)))) != value:
                raise LookupError("State is inconsistent. Hash(sha224) has changed")
        elif inspect.isfunction(glob_fkt):
            if "filename" in state and get_fkt_hash(glob_fkt) != value:
                raise LookupError("State is inconsistent. Hash(sha224) of called function '%s' has changed"%name)
        elif "filename" in state:
            raise LookupError("State is inconsistent.")
