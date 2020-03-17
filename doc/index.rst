ode-toolbox
===========

.. role:: python(code)
   :language: python

.. role:: bash(code)
   :language: bash

|Build status| |Testing coverage|

Choosing the optimal solver for systems of ordinary differential equations (ODEs) is a critical step in dynamical systems simulation. ode-toolbox is a Python package that assists in solver benchmarking, and recommends solvers on the basis of a set of user-configurable heuristics. For all dynamical equations that admit an analytic solution, ode-toolbox generates propagator matrices that allow the solution to be calculated at machine precision. For all others, the Jacobian matrix provides first-order update expressions.

The internal processing carried out by ode-toolbox can be visually summarised as follows, starting from a system of ODEs (or functions of time) on the top (double outline), and generating propagator matrices, Jacobian (first-order) update expressions, and/or recommending either a stiff or nonstiff solver (green nodes). Each step will be described below in depth.

.. raw:: html

   <img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/flow_diagram.png" alt="Flow diagram" width="620" height="463">


ode-toolbox is written in Python and leverages SymPy for the symbolic manipulation of equations. It was initially developed in the context of the `NESTML <https://github.com/nest/nestml>`__ project, in which the main focus was on the class of spiking neurons presently available in the `NEST <https://github.com/nest/nest-simulator>`__ simulator. It can, however, be used in a standalone fashion, and is broadly applicable to continuous-time dynamical systems as well as systems that undergo instantaneous events (such as neuronal spikes or impulsive forces).

Installation
------------

Prerequisites
~~~~~~~~~~~~~

Only Python 3 is supported. ode-toolbox depends on the Python packages SymPy, Cython, SciPy and NumPy (required), matplotlib and graphviz for visualisation (optional), and pytest for self-tests (also optional). The stiffness tester additionally depends on an installation of `PyGSL <http://pygsl.sourceforge.net/>`__. If PyGSL is not installed, the test for stiffness is skipped during the analysis of the equations.

All required and optional packages can be installed by running

::

    pip install -r requirements.txt

Installing ode-toolbox
~~~~~~~~~~~~~~~~~~~~~~

To install, clone the repository, go to the root directory and then run the following command in a terminal:

::

    python setup.py install

If you wish to install ode-toolbox into your home directory, add the option :bash:`--user` to the above call.

For further installation hints, please see `.travis.yml <.travis.yml>`__.

Testing
~~~~~~~

To run the unit and integration tests that come with ode-toolbox, you can run the following command:

::

    python -m pytest

Please note that this requires the `pytest <https://docs.pytest.org>`__ package to be installed.

To increase the verbosity, append the command-line parameters :bash:`-s -o log_cli=true -o log_cli_level="DEBUG"`.

Usage
-----

ode-toolbox can be used in two ways:

1. As a Python module. Import the :python:`odetoolbox` module, and then call :python:`odetoolbox.analysis(indict)` where :python:`indict` is the JSON-like input in Python dictionary format. See the tests (e.g. `test\_lorenz\_attractor.py <tests/test_lorenz_attractor.py>`__) for a full example.
2. As command line application. In this case, the input is stored in a JSON file, and ode-toolbox is invoked from the command line:

   .. code:: bash

      ./ode_analyzer.py tests/lorenz_attractor.json

The JSON file and Python dictionary are completely equivalent in content and form, described in the :ref:`Input` section below.

Several boolean flags can additionally be passed; when ode-toolbox is used via its API, these exist as function parameters (\ :python:`odetoolbox.analysis(indict, disable_stiffness_check=True, ...)`), whereas if the command line is used, they can be passed as arguments (:bash:`./ode-analyzer.py --disable_stiffness_check ...`).

.. list-table::
   :header-rows: 1
   :widths: 10 5 20

   * - Name
     - Default
     - Description
   * - ``disable_analytic_solver``
     - False
     - Set to True to return numerical solver recommendations, and no propagators, even for ODEs that are analytically tractable.
   * - ``disable_stiffness_check``
     - False
     - Set to True to disable stiffness check.
   * - ``debug``
     - False
     - Set to True to increase the verbosity.

Input
-----

The JSON input dictionary that is passed to ode-toolbox contains :ref:`dynamics <Dynamics>`, :ref:`numerical parameters <Parameters>`, and :ref:`global options <Global options>`. Documentation may optionally be provided as a string.

All expressions are parsed as SymPy expressions, and subsequently simplified through :python:`sympy.simplify()`. There are several predefined symbols, such as :python:`e` and :python:`E` for Euler's number, trigonometric functions, etc. :python:`t` is assumed to represent time. The list of predefined symbols is defined in ```shapes.py`` <odetoolbox/shapes.py>`__, as the static member :python:`Shape._sympy_globals`. Variable names should be chosen such that they do not conflict with the predefined symbols.

Dynamics
~~~~~~~~

All dynamical variables have a variable name, a differential order, and a defining expression. The overall dynamics is given as a list of these definitions. For example, we can define an alpha shape kernel :math:`g` with time constant :math:`\tau` as follows:

.. math::

   \frac{d^2g}{dt^2} = -\frac{1}{\tau^2} g - \frac{2}{\tau} \frac{dg}{dt}

This can be entered as:

.. code:: python

    "dynamics":
    [
        {
            "expression": "g'' = -1 / tau**2 * g - 2/tau * g'"
        }
    ]

Instead of a second-order differential equation, we can equivalently describe the kernel shape as a function of time:

.. math::

   g(t) = \frac{e}{\tau} t \exp(-\frac{t}{\tau})

This can be entered as:

.. code:: python

    "dynamics":
    [
        {
            "expression": "g = (e / tau) * t * exp(-t / tau)"
        }
    ]

Expressions can refer to variables defined in other expressions. For example, a third, equivalent formulation of the alpha shape is as the following system of two coupled first-order equations:

.. math::

   \frac{dg}{dt} &= h \\
   \frac{dh}{dt} &= -\frac{1}{\tau^2} g - \frac{2}{\tau} h

This can be entered as:

.. code:: python

    "dynamics":
    [
        {
            "expression": "g' = h",
        },
        {
            "expression": "h' = -g / tau**2 - 2 * h / tau",
        }
    ]


Initial values
~~~~~~~~~~~~~~

As many initial values have to be specified as the differential order requires, that is, none for functions of time, one for a one-dimensional system of ODEs, and so on. Continuing the second-order alpha function example:

.. code:: python

    "dynamics":
    [
        {
            "expression": "g'' = -1 / tau**2 * g - 2/tau * g'"
            "initial_values":
            {
                "g" : "0",
                "g'" : "e / tau"
            }
        }
    ]

If only one initial value is required, the following simpler syntax may be used, which omits the variable name:

.. code:: python

    "dynamics":
    [
        {
            "expression": "g' = -g / tau"
            "initial_value": "e / tau"
        }
    ]

Upper and lower thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~

Neuronal dynamics is typically characterised by a discontinuous jump upon action potential firing. To model this behaviour, an upper and lower bound can be defined for each input variable. When either bound is reached, the state of that variable is reset to its initial value.

Thresholds are mainly of interest when doing stiffness testing, and only apply to equations that are solved by the numerical integrator. Testing for threshold crossing and reset of the state variable(s) occurs at the beginning of every timestep.

.. code:: python

    "dynamics":
    [
        {
          "expression": "V_m' = (-g_L * (V_m - E_L) - g_ex * (V_m - E_ex)) / C_m
          "initial_value": "-70",
          "upper_bound": "-55"
        }
    }

Parameters
~~~~~~~~~~

It is not necessary to supply any numerical values for parameters. The expressions are analysed symbolically, and in some cases a set of symbolic propagators will be generated. However, in some cases (in particular when doing stiffness testing), it can be important to simulate with a particular set of parameter values. In this case, they can be specified in the global :python:`parameters` dictionary. This dictionary maps parameter names to default values, for example:

.. code:: python

    "parameters":
    {
        "N": "10",
        "C_m": "400.",
        "tau": "1 - 1/e",
        "I_ext": "30E-3"
    }

Spiking stimulus for stiffness testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spike times for each variable can be read directly from the JSON input as a list, or be generated according to a constant frequency or Poisson distribution. The general format is as follows: any number of stimuli can be defined in the global list :python:`"stimuli"`. Each entry in the list is a dictionary containing parameters, and a :python:`"variables"` attribute that specifies which dynamical variables are affected by this particular spike generator. For example:

.. code:: python

    "stimuli":
    [
        {
            "type": "poisson_generator",
            "rate": "10.",
            "variables": ["g_in'", "g_ex'"]
        }
    ]

The type is one of :python:`"poisson_generator"`, :python:`"regular"` or :python:`"list"`. The Poisson and regular spiking generators only have one parameter: rate. When the selected type is :python:`"list"`, a list of predefined spike times can be directly supplied under the key :python:`"list"`, separated by spaces, as such:

.. code:: python

    {
        "type": "list",
        "list": "5E-3 10E-3 20E-3 15E-3 50E-3",
        "variables": ["I'"]
    }

Note that the amplitude of a spike response is a result of the magnitude of its initial values.


Global options
~~~~~~~~~~~~~~

Further options for the integrator, decision criteria for solver selection and so on, can be specified in the global :python:`options` dictionary, for example:

.. code:: python

    "options" : {
        "sim_time": "100E-3",
        "max_step_size": ".25E-3"
    }

The following global options are defined. Note that all are typically formatted as strings when encoding into JSON.


.. list-table::
   :header-rows: 1
   :widths: 10 5 5 20

   * - Name
     - Default
     - Type
     - Description
   * - ``integration_accuracy_abs``
     - 1E-9
     - float
     - Absolute error bound for all numerical integrators that are used.
   * - ``integration_accuracy_rel``
     - 1E-9
     - float
     - Relative error bound for all numerical integrators that are used.
   * - ``output_timestep_symbol``
     - ``"__h"``
     - string
     - Generated propagators are a function of the simulation timestep. This parameter gives the name of the variable that contains the numerical value of the timestep during simulation.
   * - ``sim_time``
     - 100E-3
     - float
     - Total simulated time.
   * - ``max_step_size``
     - 999
     - float
     - Maximum step size during simulation (e.g. for stiffness testing solvers).
   * - ``differential_order_symbol``
     - :python:`"__d"`
     - string
     - String appended n times to output variable names to indicate differential order n. XXX: TODO: only the default value works for now.


Output
------

The analysis output is returned in the form of a Python dictionary, or an equivalent JSON file.

During analysis, ode-toolbox rewrites the differential notation from single quotation marks into characters that are typically compatible with variable names; by default every quotation mark is rewritten into the string specified as the global parameter :python:`differential_order_symbol` (by default, :python:`"__d"`).

ode-toolbox will return a list of solvers. Each solver has the following keys: 

- :python:`"solver"`\ : a string containing the solver recommendation. Starts with either :python:`"analytical"` or :python:`"numeric"`\ .
- :python:`"state_variables"`\ : an unordered list containing all variable symbols.
- :python:`"initial_values"`\ : a dictionary that maps each variable symbol (in string form) to a SymPy expression. For example :python:`"g" : "e / tau"`.
- :python:`"parameters"`\ : only present when parameters were supplied in the input. The input parameters are copied into the output for convenience.

Analytic solvers have the following extra entries:

-  :python:`"update_expressions"`\ : a dictionary that maps each variable symbol (in string form) to a SymPy propagator expression. The interpretation of an entry :python:`"g" : "g * __P__g__g + h * __P__g__h"` is that, at each integration timestep, when the state of the system needs to be updated from the current time :math:`t` to the next step :math:`t + \Delta t`, we assign the new value :python:`"g * __P__g__g + h * __P__g__h"` to the variable :python:`g`. Note that the expression is always evaluated at the old time :math:`t`; this means that when more than one state variable needs to be updated, all of the expressions have to be calculated before updating any of the variables.
-  :python:`propagators`\ : a dictionary that maps each propagator matrix entry to its defining expression; for example :python:`"__P__g__h" : "__h*exp(-__h/tau)"`

Numeric solvers have the following extra entries:

- :python:`"update_expressions"`\ : a dictionary that maps each variable symbol (in string form) to a SymPy expression that is its Jacobian, that is, for a symbol :math:`x`, the expression is equal to :math:`\frac{\delta x}{\delta t}`.


Analytic solver selection criteria
----------------------------------

If an ODE is homogeneous, constant-coefficient and linear, an analytic solution can be computed. Analytically solvable ODEs can also contain dependencies on other analytically solvable ODEs, but an otherwise analytically tractable ODE cannot depend on an ODE that can only be solved numerically. In the latter case, no analytic solution will be computed.

For example, consider an integrate-and-fire neuron with two alpha-shaped kernels (``I_shape_in`` and ``I_shape_gap``), and one nonlinear kernel (``I_shape_ex``). Each of these kernels can be expressed as a system of ODEs containing two variables. ``I_shape_in`` is specified as a second-order equation, whereas ``I_shape_gap`` is explicitly given as a system of two coupled first-order equations, i.e. as two separate ``dynamics`` entries with names ``I_shape_gap1`` and ``I_shape_gap2``.

Both formulations are mathematically equivalent, and ode-toolbox treats them the same following input processing.

During processing, a dependency graph is generated, where each node corresponds to one dynamical variable, and an arrow from node *a* to *b* indicates that *a* depends on the value of *b*. Boxes enclosing nodes mark input shapes that were specified as either a direct function of time or a higher-order differential equation, and were expanded to a system of first-order ODEs.

.. raw:: html

   <img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/eq_analysis_0.png" alt="Dependency graph" width="620" height="283">


Each variable is subsequently marked according to whether it can, by itself, be analytically solved. This is indicated by a green colour.

.. raw:: html

   <img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/eq_analysis_1.png" alt="Dependency graph with membrane potential and excitatory and gap junction kernels marked green" width="720" height="383">


In the next step, variables are unmarked as analytically solvable if they depend on other variables that are themselves not analytically solvable. In this example, ``V_abs`` is unmarked as it depends on the nonlinear excitatory kernel.

.. raw:: html

   <img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/eq_analysis_2.png" alt="Dependency graph with membrane potential and excitatory and gap junction kernels marked green" width="720" height="383">


The analytic solution for all green nodes is computed in the form of a propagator matrix. See the section :ref:"Analytic solver generation" for more details.

Numeric solver selection criteria
---------------------------------

Numeric solvers are automatically benchmarked on solving the provided system of ODEs, at a certain requested tolerance. Selecting the optimal solver is based on a set of rules, defined in :python:`StiffnessTester.draw_decision()`. The logic is as follows.

Let the machine precision (defined as the smallest representable difference between any two floating-point numbers) be written as :math:`\varepsilon`.

Then the minimum permissible timestep is defined as ``machine_precision_dist_ratio`` :math:`\cdot\varepsilon`.

-  If the minimum step size recommended by all solvers is smaller than the minimum permissible timestep, a warning is issued.
-  If the minimum step size for the implicit solver is smaller than the minimum permissible timestep, recommend the explicit solver.
-  If the minimum step size for the explicit solver is smaller than the minimum permissible timestep, recommend the implicit solver.
-  If the average step size for the implicit solver is at least ``avg_step_size_ratio`` times as large as the average step size for the explicit solver, recommend the implicit solver.
-  Otherwise, recommend the explicit solver.

.. list-table::
   :header-rows: 1
   :widths: 10 5 20

   * - Name
     - Default
     - Description
   * - ``avg_step_size_ratio``
     - 6
     - Ratio between average step sizes of implicit and explicit solver. Larger means that the explicit solver is more likely to be selected.
   * - ``machine_precision_dist_ratio``
     - 10
     - Disqualify a solver if its minimum step size comes closer than this ratio to the machine precision.


Internal representation
-----------------------

For users who want to modify/extend ode-toolbox.

Initially, individual expressions are read from JSON into Shape instances. Subsequently, all shapes are combined into a SystemOfShapes instance, which summarises all provided dynamical equations in the canonical form :math:`\mathbf{x}' = \mathbf{Ax} + \mathbf{C}`, with matrix :math:`\mathbf{A}` containing the linear part of the system dynamics and vector :math:`\mathbf{C}` containing the nonlinear terms.

Converting direct functions of time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The aim is to find a representation of the form :math:`a_0 f + a_1 f' + ... + a_{n-1} f^{(n-1)} = f^{(n)}`, with :math:`a_i\in\mathcal{R}\forall 0 \leq i < n`. The approach taken here [Blundell et al. 2018] works by evaluating the function ``f`` at times ``t = t_0, t_1, ... t_n``, which results in ``n`` equations, that we can use to solve for the coefficients of the potentially n-dimensional dynamical system.

1. Begin by assuming that the dynamical system is of order :math:`n`.
2. Find timepoints :math:`t = t_0, t_1, ..., t_n` such that :math:`f(t_i) \neq 0 \forall 0 \leq i \leq n`. The times can be selected at random.
3. Formulate the equations as :math:`\mathbf{X} \cdot \left[\begin{matrix}a_0\\a_1\\\vdots\\a_{n-1}\end{matrix}\right] = \begin{matrix}f^{(n)}(t_0)\\f^{(n)}(t_1)\\\vdots\\f^{(n)}(t_n)\end{matrix}` with :math:`\mathbf{X} = \begin{matrix}                                                    f(t_0) &  \cdots   & f^(n-1)(t_0) \\                                                     f(t_1) &  \cdots   & f^(n-1)(t_1) \\                                                     \vdots &           & \vdots \\                                                     f(t_n) &  \cdots   & f^(n-1)(t_n)                                             \end{matrix}`.
4. If :math:`\mathbf{X}` is invertible, the equation can be solved for :math:`a_0\ldots a_{n-1}`.
5. If :math:`\mathbf{X}` is not invertible, increase :math:`n` (up to some predefined maximum order :math:`n_{max}`). If :math:`n_{max}` is reached, fail.

This algorithm is implemented in :python:`Shape.from_function()` in `shapes.py <odetoolbox/shapes.py>`__.

Analytic solver generation
--------------------------

The propagator matrix :math:`P` is derived from the system matrix by matrix exponentiation:

.. math::

   P = \exp(A \cdot h)

If the imaginary unit :math:`i` is found in any of the entries in :math:`P`, fail. This usually indicates an unstable (diverging) dynamical system. Double-check the dynamical equations.

In some cases, elements of :math:`P` may contain fractions that have a factor of the form :python:`param1 - param2` in their denominator. If at a later stage, the numerical value of :python:`param1` is chosen equal to that of :python:`param2`, a numerical singularity (division by zero) occurs. To avoid this issue, it is necessary to eliminate either :python:`param1` or :python:`param2` in the input, before the propagator matrix is generated.


Working with large expressions
------------------------------

In several places during processing, a SymPy expression simplification (\ :python:`simplify()`\ ) needs to be performed to ensure correctness. For very large expressions, this can result in long wait times, while it is most often found that the resulting system of equations has no analytical solution anyway. To address these performance issues with SymPy, we introduce the :python:`Shape.EXPRESSION_SIMPLIFICATION_THRESHOLD` constant, which causes expressions whose string representation is longer than this number of characters to be skipped when simplifying expressions. The default value is 1000.


Examples
--------

Several example input files can be found under ``tests/*.json``. Some highlights:

-  `Lorenz attractor <tests/test_lorenz_attractor.json>`__
-  `Morris-Lecar neuron model <tests/morris_lecar.json>`__
-  `Integrate-and-fire neuron with alpha-kernel postsynaptic currents <tests/mixed_analytic_numerical_with_stiffness.json>`__, including Poisson spike generator for stiffness test
-  `Integrate-and-fire neuron with alpha-kernel postsynaptic conductances <tests/iaf_cond_alpha_odes_stiff.json>`__
-  `Canonical, two-dimensional stiff system <tests/stiff_system.json>`__ Example 11.57 from Dahmen, W., and Reusken, A. (2005). Numerik fuer Naturwissenschaftler. Berlin: Springer


Stiffness testing
~~~~~~~~~~~~~~~~~

This example correponds to the unit test in `test_stiffness.py <tests/test_stiffness.py>`_, which simulates the Morris-Lecar neuron model in `morris_lecar.json <tests/morris_lecar.json>`_. The plot shows the two state variables of the model, ``V`` and ``W``, while in the lower panel the solver timestep recommendation is plotted at each step. This recommendation is returned by each GSL solver. Note that the ``avg_step_size_ratio`` selection criterion parameter refers to the *average* of this value across the entire simulation period.

.. raw:: html

   <img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/stiffness_example.png" alt="timeseries plots of V, W, and recommended timestep" width="620" height="434">


`test_stiffness.py <tests/test_stiffness.py>`_ tests that for a tighter integration accuracy, the solver recommendation for this example changes from "explicit" (non-stiff) to "implicit" (stiff).

From ode-toolbox results dictionary to simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ode-toolbox provides two classes that can perform numerical simulation on the basis of the results dictionary returned by ode-toolbox: :python:`AnalyticIntegrator` (in `analytic_integrator.py <odetoolbox/analytic_integrator.py>`_, which simulates on the basis of propagators and returns precise values, and :python:`MixedIntegrator` (in `mixed_integrator.py <odetoolbox/mixed_integrator.py>`_\ ), which in addition performs numerical integration using GSL (for example, using :python:`pygsl.odeiv.step_rk4` or :python:`pygsl.odeiv.step_bsimp`). These integrators both use :python:`sympy.parsing.sympy_parser` to parse the expression strings from the ode-toolbox results dictionary, and then use the SymPy expression :python:`evalf()` method to evaluate to a floating-point value.

The file `test_analytic_solver_integration.py <tests/test_analytic_solver_integration.py>`_ contains an integration test that uses :python:`AnalyticIntegrator` and the propagators returned from ode-toolbox to simulate a simple dynamical system; in this case, an integrate-and-fire neuron with alpha-shaped postsynaptic currents. It compares the obtained result to a handwritten solution, which is simulated analytically and numerically independent of ode-toolbox. The following results figure shows perfect agreement between the three simulation methods:

.. raw:: html

   <img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/test_analytic_solver_integration.png" alt="V_abs, i_ex and i_ex' timeseries plots" width="620" height="465">


The file `test_mixed_integrator_numeric.py <tests/test_mixed_integrator_numeric.py>`_ contains an integration test, that uses :python:`MixedIntegrator` and the results dictionary from ode-toolbox to simulate the same integrate-and-fire neuron with alpha-shaped postsynaptic response, but purely numerically (without the use of propagators). In contrast to the :python:`AnalyticIntegrator`, enforcement of upper- and lower bounds is supported, as can be seen in the behaviour of :math:`V_m` in the plot that is generated:

.. raw:: html

   <img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/test_mixed_integrator_numeric.png" alt="g_in, g_in__d, g_ex, g_ex__d, V_m timeseries plots" width="620" height="451">


Caching of results
------------------

.. admonition:: TODO

   Not implemented yet

Some operations on SymPy expressions can be quite slow (see the section :ref:`Working with large expressions`\ ).

Even dynamical systems of moderate size can require a few minutes of processing time, in large part due to SymPy calls, and solver selection.

To speed up processing, a caching mechanism analyses the final system matrix :math:`A` and rewrites it as a block-diagonal matrix :math:`A = \text{diag}(B_1, B_2, \dots, B_k)`, were each of :math:`B_1, B_2, \dots, B_k` is square.

For propagators, we note that

.. math::

   e^{At} = \text{diag}(e^{B\_1t}, e^{B\_2t}, \dots, e^{B\_kt})


API documentation
-----------------

The documentation of classes and functions in the odetoolbox Python module can be found here: :mod:`odetoolbox`


Contributions and getting help
------------------------------

The primary development of ode-toolbox happens on GitHub, at https://github.com/nest/ode-toolbox. If you encounter any issue, please create an new entry in the GitHub issue tracker. Pull requests are welcome.


Citing ode-toolbox
------------------

If you use ode-toolbox in your work, please cite it as:

.. admonition:: TODO

   Will insert the Zenodo reference here to ode-toolbox once released.


References
----------

1. Inga Blundell, Dimitri Plotnikov, Jochen Martin Eppler and Abigail Morrison (2018) **Automatically selecting a suitable integration scheme for systems of differential equations in neuron models.** Front. Neuroinform. `doi:10.3389/fninf.2018.00050 <https://doi.org/10.3389/fninf.2018.00050>`__. Preprint available on `Zenodo <https://zenodo.org/record/1411417>`__.


Acknowledgements
----------------

This software was initially supported by the JARA-HPC Seed Fund *NESTML - A modeling language for spiking neuron and synapse models for NEST* and the Initiative and Networking Fund of the Helmholtz Association and the Helmholtz Portfolio Theme *Simulation and Modeling for the Human Brain*.

This software was developed in part or in whole in the Human Brain Project, funded from the European Union's Horizon 2020 Framework Programme for Research and Innovation under Specific Grant Agreements No. 720270 and No. 785907 (Human Brain Project SGA1 and SGA2).

.. |Build status| image:: https://travis-ci.org/nest/ode-toolbox.svg?branch=master
   :target: https://travis-ci.org/nest/ode-toolbox
.. |Testing coverage| image:: https://codecov.io/gh/nest/ode-toolbox/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/nest/ode-toolbox
