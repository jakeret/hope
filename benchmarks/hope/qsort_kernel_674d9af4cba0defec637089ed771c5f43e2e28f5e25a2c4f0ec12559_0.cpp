
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

inline std::tuple<PyObject *, npy_intp const *, npy_double *> qsort_kernel_d1JJ(
	  PyObject * pa, npy_intp const * __restrict__ sa, npy_double * __restrict__ ca
	, npy_int64 clo
	, npy_int64 chi
);
inline std::tuple<PyObject *, npy_intp const *, npy_double *> qsort_kernel_d1JJ(
	  PyObject * pa, npy_intp const * __restrict__ sa, npy_double * __restrict__ ca
	, npy_int64 clo
	, npy_int64 chi
) {
	npy_int64 ci = npy_int64();
	ci = clo;
	npy_int64 cj = npy_int64();
	cj = chi;
	if (false) {
		return std::make_tuple((PyObject *)pa, sa, ca);
	}
	while ((npy_bool)(ci < chi)) {
		npy_double cpivot = npy_double();
		cpivot = ca[(int)((npy_int64)((npy_int64)((npy_int64)clo + (npy_int64)chi) / (npy_int64)2))];
		while ((npy_bool)(ci <= cj)) {
			while ((npy_bool)(ca[(int)(ci)] < cpivot)) {
				ci += (npy_int64)1;
			}
			while ((npy_bool)(ca[(int)(cj)] > cpivot)) {
				cj -= (npy_int64)1;
			}
			if ((npy_bool)(ci <= cj)) {
				npy_double ctmp = npy_double();
				ctmp = ca[(int)(ci)];
				ca[(int)(ci)] = ca[(int)(cj)];
				ca[(int)(cj)] = ctmp;
				ci += (npy_int64)1;
				cj -= (npy_int64)1;
			}
		}
		if ((npy_bool)(clo < cj)) {
			qsort_kernel_d1JJ(pa, sa, ca, clo, cj);
		}
		clo = ci;
		cj = chi;
	}
	return std::make_tuple((PyObject *)pa, sa, ca);
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
			PyObj pa;
			PyObject * plo; npy_int64 clo;
			PyObject * phi; npy_int64 chi;
			if (
				PyTuple_CheckExact(args) and PyTuple_GET_SIZE(args) == 3
				and (pa = PyTuple_GET_ITEM(args, 0)) and PyArray_CheckExact(pa)
				and PyArray_TYPE((PyArrayObject *)pa) == NPY_FLOAT64 and PyArray_NDIM((PyArrayObject *)pa) == 1
				and (plo = PyTuple_GET_ITEM(args, 1)) and PyLong_CheckExact(plo)
				and (phi = PyTuple_GET_ITEM(args, 2)) and PyLong_CheckExact(phi)
			) {
				if (!(pa.incref((PyObject *)PyArray_GETCONTIGUOUS((PyArrayObject *)pa)))) {
					PyErr_SetString(PyExc_ValueError, "Invalid Argument type on a!");
					return NULL;
				}
				clo = PyLong_AS_LONG(plo);
				chi = PyLong_AS_LONG(phi);
				try {
					PyObject * res = std::get<0>(qsort_kernel_d1JJ(
						  pa, PyArray_SHAPE((PyArrayObject *)pa), (npy_double *)PyArray_DATA((PyArrayObject *)pa)
						, clo
						, chi
					));

					Py_INCREF(res);
					return res;
				} catch (...) {
					return NULL;
				}
			} else
				PyErr_Clear();
		}
		PyObject * signatures = Py_BuildValue("(sO)", "gANdcQBdcQEoY2hvcGUuX2FzdApWYXJpYWJsZQpxAimBcQN9cQQoWAQAAABuYW1lcQVYAQAAAGFx\nBlgFAAAAZHR5cGVxB1gHAAAAZmxvYXQ2NHEIWAQAAABkaW1zcQlLAXViaAIpgXEKfXELKGgFWAIA\nAABsb3EMaAdjYnVpbHRpbnMKaW50CnENaAlLAHViaAIpgXEOfXEPKGgFWAIAAABoaXEQaAdoDWgJ\nSwB1YmVhLg==\n", args);
		if (!signatures) {
			PyErr_SetString(PyExc_ValueError, "Error building signature string for qsort_kernel");
			return NULL;
		}
		return PyObject_Call(create_signature, signatures, NULL);
	}

    PyMethodDef qsort_kernelMethods[] = {
        { "set_create_signature", set_create_signature, METH_VARARGS, "signal handler" },
        { "run", run, METH_VARARGS, "module function" },
        { NULL, NULL, 0, NULL }
    };
    

    static struct PyModuleDef qsort_kernelmodule = {
        PyModuleDef_HEAD_INIT,
        "qsort_kernel",
        NULL,
        -1,
        qsort_kernelMethods
    };
    

    PyMODINIT_FUNC PyInit_qsort_kernel_674d9af4cba0defec637089ed771c5f43e2e28f5e25a2c4f0ec12559_0(void) {
            import_array();
            PyImport_ImportModule("numpy");
            return PyModule_Create(&qsort_kernelmodule);
    }
    
}
