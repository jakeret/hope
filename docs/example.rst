Example
=======


Assume the following example: 

poly.py
-------

.. code-block:: python

    from hope import jit

    def poly(x, y, a):
    	x1 = x - a
    	y[:] = x1 + x1 * x1

    poly_hope = jit(poly)


Step-by-step evaluation
-----------------------

In the following we analyze the execution of the example.

call1.py
^^^^^^^^
.. code-block:: python

    from poly import poly
    import numpy as np

    y = np.empty(1000, dtype=np.float32)
    poly_hope(np.random.random(1000).astype(np.float32), y, 3.141)

Executing ``python call1.py`` will cause the following steps to happen:

When evaluating the statement ``poly_hope = jit(poly)``

	#. **HOPE** checks if a shared object of a compiled version of ``poly`` is available. Since we run it the first time no object is available, so **HOPE** returns a wrapper function that contains a reference to the original function.

When evaluating the statement ``poly_hope(np.random.random(1000).astype(np.float32), y, 3.141)``

	#. The wrapper function, which was returned by jit, is called
	
	#. A Python AST of ``poly`` is generated:

		.. code-block:: none

			FunctionDef(
			      name='poly'
			    , args=arguments(args=[Name(id=x), Name(id=y), Name(id=a)])
			    , body=[
			          Assign(
			              targets=[Name(id=x1)]
			            , value=BinOp(left=Name(id=x), op=Sub, right=Name(id=a))
			          )
			        , Assign(
			              targets=[Subscript(value=Name(id=y) 
			                                 , slice=Slice(lower=None, upper=None, step=None))]
			           , value=BinOp(left=Name(id=x1)
			                         , op=Add 
			                         , right=BinOp(left=Name(id=x1) 
			                                     , op=Mult
			                                     , right=Name(id=x1)))
			          )
			      ]
			)


	#. The arguments passed to ``poly`` are analyzed:
	
		- ``x``: ``numpy.float32``, 1D
		- ``y``: ``numpy.float32``, 1D
		- ``a``: ``numpy.float64``, scalar (originally ``a`` has type ``float`` but this is equivalent to ``numpy.float64``)

	#. **HOPE** generates an identification for the arguments: ``f1f1d``

	#. **HOPE** generates a HOPE AST from the Python AST and the analyzed arguments:

		.. code-block:: none

			Module(
			      main=poly
			    , functions=[
			        FunctionDef(
			              name='poly'
			            , args=arguments(args=[
			                  Variable(id=x, shape=(0, x_0), dtype=numpy.float32 
			                           , scope=signature, allocated=true)
			                , Variable(id=y, shape=(0, y_0), dtype=numpy.float32 
			                           , scope=signature, allocated=true)
			                , Variable(id=a, shape=(), dtype=numpy.float64 
			                           , scope=signature, allocated=true)
			              ])
			            , merged=[[(0, x_0), (0, y_0)]]
			            , body=[
			                  Block(body=[
			                      Assign(
			                          target=Variable(id=x1, shape=(0, x_0), dtype=numpy.float32
			                                        , scope=block, allocated=false)
			                        , value=BinOp(left=Variable(id=x, ...)
			                                    , op=Sub, right=Variable(id=a, ...)
			                                    , shape=(0, x_0), dtype=numpy.float32)
			                    , Assign(
			                          target=View(variable=Variable(id=y, ...)
			                                    , extend=[0, y_0)
			                                    , shape=(0, x_0), dtype=numpy.float32
			                        , value=BinOp(
			                              left=Variable(id=x, ...)
			                            , op=Sub
			                            , right=BinOp(left=Variable(id=x1, ...)
			                                        , op=Mult, right=Variable(id=x1, ...)
			                                        , shape=(0, x_0), dtype=numpy.float32)
			                            , shape=(0, x_0), dtype=numpy.float32
			                          )
			                  ], shape=(0, x_0), dtype=numpy.float32)
			              ]
			        )			
			      ]
			)

		Differences between the Python AST and the HOPE AST:

		* The **HOPE** AST is statically typed, each token has a scalar type (``dtype``) and for a start, stop for each dimension (``shape``) where shape=(0, x_0) means start=0, stop=x.shape[0]

		* The function definition has a property *merged*. This list of lists identifies all ``segments`` (each dimension of a ``shape`` is called ``segment``), which are equal. This is determined as follow:
			
			- the statement ``x1 = x - a`` implies that ``x1`` has the same shape as ``x``
			- the statement zz ``y[:] = x1 + x1 * x1`` is only valid if x1 and y have the same shape.

			so ``x`` and ``y`` must have the same shape.

		* The function body contains a ``Block`` token. This token is generated the following way:

			#. Each statement in the body is wrapped into a ``Block`` token. Each ``Block`` token has the shape of the statement
			#. All neighbor blocks with the same shape are merged

		* Variables have a scope, which can either be:
		
			- ``signature``: variables that are passed on call
			- ``body``: variables, which occur in more than one Block
			- ``block``: variables, which occur only in one Block token

	#. **HOPE** traverses the new AST in order to identify numerical optimization possibilities :ref:`optimization`

	#. generate C++11 code from the **HOPE** AST. The Block taken above is translated into the following C++ code:

		- the shape of ``x`` is stored in the ``sx`` array
		- the C pointer to the data of ``x`` is stored ``cx``, ``ca`` is a double value containing the value of ``a``

		.. code-block:: c++

		    for (npy_intp i0 = 0; i0 < sx[0] - 0; ++i0) {
		    	auto cx1 = (cx[i0] - ca);
		    	cy[i0] = (cx1 + (cx1 * cx1));
		    }

		- The whole ``Block`` statement is turned into one loop over the shape of the block. This allows us to evaluate the operation element-wise, which improves cache locality.

		- For variables with ``Block`` scope there is no need to allocate a whole array, we only allocate a scalar value.

	#. the C++ code is compiled into a shared object library

	#. the shared object library is dynamically imported and the compiled function is evaluated.

call2.py
^^^^^^^^
.. code-block:: python

    from poly import poly
    import numpy as np

    y = np.empty(1000, dtype=np.float32)
    poly_hope(np.random.random(1000).astype(np.float32), y, 3.141)

    y = np.empty(1000, dtype=np.float64)
    poly_hope(np.random.random(1000).astype(np.float64), y, 42)

Executing ``python call2.py`` will cause the following steps to happen:

When evaluating the statement ``poly_hope = jit(poly)``
	
	#. checks if a shared object of a compiled version of ``poly`` is available. Since a shared object is available the shared object is dynamically loaded

	#. a callback function for unknown signatures is registered in the module

	#. the reference to the compiled ``poly`` function is returned

When evaluating the statement ``poly_hope(np.random.random(1000).astype(np.float32), y, 3.141)``

	#. the compiled ``poly`` function is called

When evaluating the statement ``poly_hope(np.random.random(1000).astype(np.float64), y, 42)``

	#. there is no compiled ``poly`` function for the passed argument types, so the registered callback is called

	#. the arguments which are passed to ``poly`` are analysed:

		- ``x``: ``numpy.float64``, 1D
		- ``y``: ``numpy.float64``, 1D
		- ``a``: ``numpy.int64``, scalar (originally ``a`` has type ``int`` but this is equivalent to ``numpy.int64``)

	#. The code is regenerated as described above, but this time with two different function signatures. Once for 

		- ``x``: ``numpy.float32``, 1D
		- ``y``: ``numpy.float32``, 1D
		- ``a``: ``numpy.float64``, scalar

		and once for 

		- ``x``: ``numpy.float64``, 1D
		- ``y``: ``numpy.float64``, 1D
		- ``a``: ``numpy.int64``, scalar (originally ``a`` has type ``int`` but this is equivalent to ``numpy.int64``)

	#. The new shared object library is dynamically imported and evaluated

