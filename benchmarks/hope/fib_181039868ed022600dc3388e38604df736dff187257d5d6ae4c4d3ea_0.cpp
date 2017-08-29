
#define PY_ARRAY_UNIQUE_SYMBOL fkt_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <cmath>
#include <tuple>
#include <numeric>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <type_traits>


struct PyObj {
    typedef PyObject * ptr_t;
    typedef PyArrayObject * arrptr_t;
    PyObj(): dec(false), ptr(NULL) {}
    PyObj(ptr_t p): dec(false), ptr(p) {}
    ~PyObj() { if(dec) Py_DECREF(ptr); }
    PyObj & operator=(ptr_t p) { if(dec) Py_DECREF(ptr); ptr = p; dec = false; return *this; }
    PyObj & incref(ptr_t p) { if(dec) Py_DECREF(ptr); ptr = p; dec = (p != NULL); return *this; }
    operator bool() const { return ptr; }
    operator ptr_t() const { return ptr; }
    operator arrptr_t() const { return (arrptr_t)ptr; }
    bool dec;
    ptr_t ptr;
};

inline npy_int64 fib_J(
	  npy_int64 cn
);
inline npy_int64 fib_J(
	  npy_int64 cn
) {
	if ((npy_bool)(cn < (npy_int64)2)) {
		return cn;
	}
	return (npy_int64)((npy_int64)fib_J((npy_int64)((npy_int64)cn - (npy_int64)(npy_int64)1)) + (npy_int64)fib_J((npy_int64)((npy_int64)cn - (npy_int64)(npy_int64)2)));
	PyErr_SetString(PyExc_ValueError, "No return type passed!");
	throw std::exception();
}

#include <string>
#include <sstream>
#include <iostream>
#include <cxxabi.h>
#include <execinfo.h>
#include <signal.h>

void sighandler(int sig);

void sighandler(int sig) {
    std::ostringstream buffer;
    buffer << "Abort by " << (sig == SIGSEGV ? "segfault" : "bus error") << std::endl;
    void * stack[64];
    std::size_t depth = backtrace(stack, 64);
    if (!depth)
        buffer << "  <empty stacktrace, possibly corrupt>" << std::endl;
    else {
        char ** symbols = backtrace_symbols(stack, depth);
        for (std::size_t i = 1; i < depth; ++i) {
            std::string symbol = symbols[i];
                if (symbol.find_first_of(' ', 59) != std::string::npos) {
                    std::string name = symbol.substr(59, symbol.find_first_of(' ', 59) - 59);
                    int status;
                    char * demangled = abi::__cxa_demangle(name.c_str(), NULL, NULL, &status);
                    if (!status) {
                        buffer << "    " 
                            << symbol.substr(0, 59) 
                            << demangled
                            << symbol.substr(59 + name.size())
                            << std::endl;
                        free(demangled);
                    } else
                        buffer << "    " << symbol << std::endl;
                } else
                    buffer << "    " << symbol << std::endl;
            }
            free(symbols);
        }
        std::cerr << buffer.str();
        std::exit(EXIT_FAILURE);
    }
    

extern "C" {

    PyObject * create_signature;
    
    struct sigaction slot;
    
    PyObject * set_create_signature(PyObject * self, PyObject * args) {
        if (!PyArg_ParseTuple(args, "O", &create_signature)) {
            PyErr_SetString(PyExc_ValueError, "Invalid Argument to set_create_signature!");
            return NULL;
        }
        Py_INCREF(create_signature);
        memset(&slot, 0, sizeof(slot));
        slot.sa_handler = &sighandler;
        sigaction(SIGSEGV, &slot, NULL);
        sigaction(SIGBUS, &slot, NULL);
        Py_INCREF(Py_None);
        return Py_None;
    }

	PyObject * run(PyObject * self, PyObject * args) {
		{
			PyObject * pn; npy_int64 cn;
			if (
				PyTuple_CheckExact(args) and PyTuple_GET_SIZE(args) == 1
				and (pn = PyTuple_GET_ITEM(args, 0)) and PyLong_CheckExact(pn)
			) {
				cn = PyLong_AS_LONG(pn);
				try {
					return Py_BuildValue("l", fib_J(
						  cn
					));
				} catch (...) {
					return NULL;
				}
			} else
				PyErr_Clear();
		}
		PyObject * signatures = Py_BuildValue("(sO)", "gANdcQBdcQFjaG9wZS5fYXN0ClZhcmlhYmxlCnECKYFxA31xBChYBQAAAGR0eXBlcQVjYnVpbHRp\nbnMKaW50CnEGWAQAAABkaW1zcQdLAFgEAAAAbmFtZXEIWAEAAABucQl1YmFhLg==\n", args);
		if (!signatures) {
			PyErr_SetString(PyExc_ValueError, "Error building signature string for fib");
			return NULL;
		}
		return PyObject_Call(create_signature, signatures, NULL);
	}

    PyMethodDef fibMethods[] = {
        { "set_create_signature", set_create_signature, METH_VARARGS, "signal handler" },
        { "run", run, METH_VARARGS, "module function" },
        { NULL, NULL, 0, NULL }
    };
    

    static struct PyModuleDef fibmodule = {
        PyModuleDef_HEAD_INIT,
        "fib",
        NULL,
        -1,
        fibMethods
    };
    

    PyMODINIT_FUNC PyInit_fib_181039868ed022600dc3388e38604df736dff187257d5d6ae4c4d3ea_0(void) {
            import_array();
            PyImport_ImportModule("numpy");
            return PyModule_Create(&fibmodule);
    }
    
}
