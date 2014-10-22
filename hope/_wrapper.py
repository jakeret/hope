# Copyright (c) 2013 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

# setuptools have a proplem with the unicode_literals in python 2, so skip unicode_literals
from __future__ import print_function, division, absolute_import


import os
import io
import sys
import hashlib
import setuptools
import pickle
import inspect
import tempfile
import shutil
import warnings

from numpy.distutils.misc_util import get_numpy_include_dirs
from setuptools.command.build_ext import build_ext
from distutils.dist import Distribution
import distutils.sysconfig


from hope._ast import *
from hope._transformer import ASTTransformer
from hope import config
import hope._cache as cache


class Wrapper:

    _recompile = False

    # the deepcopy of the ast does violates the deepcopy semantics, so do not allow deepcopy ...
    def __deepcopy__(self, memo):
        raise TypeError("The hope function wrapper can't be cloned")

    def __init__(self, fkt, hash):
        self.modtoken, self.fkt, self.filename, self.cache = Module(fkt.__name__), fkt, "{0}_{1}".format(fkt.__name__, hash), None

        def recoverArg(arg):
            if isinstance(arg, Object):
                if not hasattr(arg, "parent"):
                    arg.parent = None
                for name, value in list(arg.attrs.items()):
                    if isinstance(value, (Object, ObjectAttr)):
                        value.parent = arg
                    arg.attrs[name] = recoverArg(value)
            else:
                if isinstance(arg.dtype, str) and hasattr(np, arg.dtype):
                    arg.dtype = getattr(np, arg.dtype)
                arg.shape = [(None, Dimension(arg, dim)) for dim in range(arg.dims)]
                delattr(arg, "dims")
                if not isinstance(arg, ObjectAttr):
                    arg.scope, arg.allocated = "signature", True
            return arg

        def create_signature(pickled, args):
            if sys.version_info[0] == 2:
                unpickled = pickle.loads(pickled)
            else:
                import base64
                unpickled = pickle.loads(base64.decodebytes(pickled.encode('ascii')))
            for striped in unpickled:
                signature = [recoverArg(arg) for arg in striped]
                ASTTransformer(self.modtoken).module_visit(self.fkt, signature, "".join([arg.getId() for arg in signature]))
            return self(*args)
        self.create_signature = create_signature

        def _hope_callback(*args):
            return self(*args) if self.cache is None else self.cache(*args)
        self.callback = _hope_callback
        setattr(cache, str(id(self.callback)), fkt)


    def __call__(self, *args):
        ASTTransformer(self.modtoken).module_visit(self.fkt, args)

        if config.optimize:

            if config.verbose:
                from hope import _dump
                # TODO: use log
                print(_dump.Dumper().visit(self.modtoken))

            from hope._optimizer import Optimizer
            Optimizer().visit(self.modtoken)

        if config.verbose:
            self._print_module_info()

        try:
            tempfolder = tempfile.mkdtemp(prefix='hope')
    
            # TODO: if function is class method, make classname_method name as name
            localfilename = "{0}_{1}".format(self.filename, len(self.modtoken.functions[self.modtoken.main]) - 1)
    
            from hope._generator import generate
            code = generate(self.modtoken, localfilename)
    
            with open(os.path.join(tempfolder, "{0}.cpp".format(localfilename)), "w") as fp:
                fp.write(code)
    
            so_filename = _compile(tempfolder, localfilename, self.fkt.__name__)
    
            self._store_state(tempfolder, localfilename)
    
            #move xxx.so and pickled state if it doesn't already exists or if it was a recompilation due to inconsistent state
            if self._recompile or not os.path.isfile(os.path.join(config.prefix, "{0}.so".format(localfilename))):
                shutil.move(os.path.join(tempfolder, so_filename), os.path.join(config.prefix, "{0}.so".format(localfilename)))
                shutil.move(os.path.join(tempfolder, "{0}.pck".format(self.filename)), os.path.join(config.prefix, "{0}.pck".format(self.filename)))
    
    
            if sys.version_info[0] == 2:
                module = __import__(localfilename, globals(), locals(), [], -1)
            else:
                import importlib
                importlib.invalidate_caches()
                module = importlib.import_module(localfilename)
            module.set_create_signature(self.create_signature)
    
            for name in list(sys.modules.keys()):
                if hasattr(sys.modules[name], self.fkt.__name__) and getattr(sys.modules[name], self.fkt.__name__) is self.callback:
                    setattr(sys.modules[name], self.fkt.__name__, module.run)
            setattr(cache, str(id(module.run)), self.fkt)
            self.cache = module.run
    
            return module.run(*args)
        
        finally:
            if config.keeptemp:
                for ext in ["out", "cpp", "o"]:
                    if os.path.isfile(os.path.join(tempfolder, "{0}.{1}".format(localfilename, ext))):
                        shutil.move(os.path.join(tempfolder, "{0}.{1}".format(localfilename, ext)), os.path.join(config.prefix, "{0}.{1}".format(localfilename, ext)))

            shutil.rmtree(tempfolder)
            

    def _store_state(self, tempfolder, localfilename):
        state = {"filename": localfilename, 
                 "main": self.modtoken.main, 
                 "called": {}, 
                 "bound": any([main.isbound for main in self.modtoken.functions[self.modtoken.main]]) 
                 }
        
        for name in list(self.modtoken.functions.keys()):
            if name != self.fkt.__name__:
                fkt_hash = None
                # distinguish if a function does not exists or if its an object method
                if not name in self.fkt.__globals__: pass
                #     raise Exception("Function not accessible form global scope of function: {0} ({1})".format(self.fkt.__name__, name))
                # TODO: remove this!
                elif isinstance(self.fkt.__globals__[name], Wrapper):
                    fkt_hash = get_fkt_hash(self.fkt.__globals__[name].fkt)
                elif inspect.isbuiltin(self.fkt.__globals__[name]) and hasattr(cache, str(id(self.fkt.__globals__[name]))):
                    fkt_hash = get_fkt_hash(getattr(cache, str(id(self.fkt.__globals__[name]))))
                elif inspect.isfunction(self.fkt.__globals__[name]):
                    fkt_hash = get_fkt_hash(self.fkt.__globals__[name])
                else:
                    raise Exception("Function not a unbound, pure python function: {0} ({1})".format(self.fkt.__name__, name))
                
                if fkt_hash is not None:
                    state["called"][name] = fkt_hash

        for name in get_config_attrs():
            state[name] = getattr(config, name)

        with open(os.path.join(tempfolder, "{0}.pck".format(self.filename)), "wb") as fp:
            pickle.dump(state, fp)

    def _print_module_info(self):
#         from hope import _dump
        # TODO: use log
#             print(_dump.Dumper().visit(self.modtoken))
        print("Compiling following functions:")
        for (k,v) in self.modtoken.functions.items():
            for node in v:
                signature = []
                for arg in node.signature:
                    if isinstance(arg, Variable):
                        signature.append("{0}{1!s} {2}".format(NPY_TYPE[arg.dtype], "" if len(arg.shape) == 0 else "^{0}".format(len(arg.shape)), arg.name))
                    elif isinstance(arg, Object):
                        signature.append("{0} {1}".format(arg.classname, arg.name))
                    else:
                        raise Exception("Unknown signature argument: {0!s}".format(self.visit(arg)))
                print("{0}({1})".format(node.name, ", ".join(signature)))
            

def _compile(target, localfilename, fkt_name):
    """
    Compiles a C++ function into a shared object library
    
    :param target: target folder where the .cpp file is located and where the output should be stored
    :param localfilename: name of file to be compiled without '.cpp' suffix
    :param fkt_name: name of the function to be compiled
    
    :raises Exception: An exception is raised if the file could be to compiled
    
    :return so_filename: The name of the .so file
    """
    outfile, stdout, stderr, argv = None, None, None, sys.argv
    try:
        sys.stdout.flush(), sys.stderr.flush()
        outfile = open(os.path.join(target, "{0}.out".format(localfilename)), 'w')

        if sys.version_info[0] == 2 and isinstance(sys.stdout, file) and isinstance(sys.stdout, file):
            stdout, stderr = os.dup(sys.stdout.fileno()), os.dup(sys.stderr.fileno())
            os.dup2(outfile.fileno(), sys.stdout.fileno())
            os.dup2(outfile.fileno(), sys.stderr.fileno())
        else:
            stdout, stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = outfile, outfile
        try:
            sources = os.path.join(target, "{0}.cpp".format(localfilename))
            sys.argv = ["", "build_ext",
                        "-b", target,  #--build-lib (-b)     directory for compiled extension modules
                        "-t", "/" #--build-temp - a rel path will result in a dir structure of -b at the cur position 
                        ]

            # setuptools have a problem with the unicode_literals in python 2 ...
            if sys.version_info[0] == 2:
                from types import StringType
                localfilename = StringType(localfilename)
                sources = StringType(sources)

            # avoid warning on linux + gcc
            cfg_vars = distutils.sysconfig.get_config_vars()
            if "CFLAGS" in cfg_vars:
                cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")
            if "OPT" in cfg_vars:
                cfg_vars["OPT"] = cfg_vars["OPT"].replace("-Wstrict-prototypes", "")

            setuptools.setup( \
                  name = localfilename\
                , ext_modules = [setuptools.Extension( \
                      localfilename \
                    , sources = [sources] \
                    , extra_compile_args = config.cxxflags \
                  )] \
                , include_dirs = get_numpy_include_dirs() \
            )
        except SystemExit as e:
            print(sys.stderr.write(str(e)))
        sys.stdout.flush(), sys.stderr.flush()
    finally:
        if isinstance(stdout, int):
            os.dup2(stdout, sys.stdout.fileno()), os.close(stdout)
        elif not stdout is None:
            sys.stdout = stdout
        if isinstance(stderr, int):
            os.dup2(stderr, sys.stderr.fileno()), os.close(stderr)
        elif not stderr is None:
            sys.stderr = stderr
        if (sys.version_info[0] == 2 and isinstance(outfile, file))\
                or (sys.version_info[0] == 3 and isinstance(outfile, io.TextIOWrapper) and not outfile.closed):
            outfile.close()
        sys.argv = argv

    with open(os.path.join(target, "{0}.out".format(localfilename))) as outfile:
        out = outfile.read()

    # on Travis CI & Py33 name contains additional suffix to .so
    so_filename = build_ext(Distribution()).get_ext_filename(localfilename)

    if not os.path.isfile(os.path.join(target, so_filename)) or out.find("error:") > -1:
        print(out)
        raise Exception("Error compiling function {0} (compiled to {1})".format(fkt_name, target))
    
    #TODO: add test case
    if out.find("warning:") > -1:
        try: 
            #trying to encode utf-8 to support AstroPy
            warnings.warn("A warning has been issued during compilation:\n{0}".format(out).encode('utf-8'))
        except UnicodeError:
            #encoding fails on Linux
            warnings.warn("A warning has been issued during compilation:\n{0}".format(out))

    if config.verbose:
        print(out)
        
    return so_filename

def get_config_attrs():
    """
    Returns the attributes of the hope config, filtering private attributes and imports from __future__
    
    :return generator: a generator yielding the attributes
    """
    return (name for name in dir(config) if (not name.startswith("__") and name not in ("print_function", "division", "absolute_import", "unicode_literals")))
    
    
def get_fkt_hash(fkt):
    """
    Returns hash of the `uft-8` encoded source of the given function
    
    :param fkt: the function to compute the hash from
    
    :return hash: the sha224 hash code 
    """
    return hashlib.sha224(inspect.getsource(fkt).encode('utf-8')).hexdigest()
