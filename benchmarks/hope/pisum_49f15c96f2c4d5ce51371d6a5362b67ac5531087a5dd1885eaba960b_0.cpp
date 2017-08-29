
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

inline npy_double pisum_(
	  );
inline npy_double pisum_(
	  ) {
	npy_double csum = npy_double();
	for (npy_intp cj = (npy_int64)1; cj < (npy_int64)501; ++cj) {
		csum = (npy_double)0.0;
		auto cf = (npy_double)0.0;
		for (npy_intp ck = (npy_int64)1; ck < (npy_int64)10001; ++ck) {
			auto c__sp0 = (npy_int64)((npy_int64)ck * (npy_int64)ck);
			csum += (npy_double)((npy_double)(npy_double)1.0 / (npy_double)c__sp0);
		}
	}
	return csum;
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
		{				try {
					return Py_BuildValue("d", pisum_());
				} catch (...) {
					return NULL;
				}
		}
		PyObject * signatures = Py_BuildValue("(sO)", "gANdcQBdcQFhLg==\n", args);
		if (!signatures) {
			PyErr_SetString(PyExc_ValueError, "Error building signature string for pisum");
			return NULL;
		}
		return PyObject_Call(create_signature, signatures, NULL);
	}

    PyMethodDef pisumMethods[] = {
        { "set_create_signature", set_create_signature, METH_VARARGS, "signal handler" },
        { "run", run, METH_VARARGS, "module function" },
        { NULL, NULL, 0, NULL }
    };
    

    static struct PyModuleDef pisummodule = {
        PyModuleDef_HEAD_INIT,
        "pisum",
        NULL,
        -1,
        pisumMethods
    };
    

    PyMODINIT_FUNC PyInit_pisum_49f15c96f2c4d5ce51371d6a5362b67ac5531087a5dd1885eaba960b_0(void) {
            import_array();
            PyImport_ImportModule("numpy");
            return PyModule_Create(&pisummodule);
    }
    
}
