# NEST ode-toolbox

[![Build status](https://travis-ci.org/nest/ode-toolbox.svg?branch=master)](https://travis-ci.org/nest/ode-toolbox) [![Testing coverage](https://codecov.io/gh/nest/ode-toolbox/branch/master/graph/badge.svg)](https://codecov.io/gh/nest/ode-toolbox)

Choosing the optimal solver for systems of ordinary differential equations (ODEs) is a critical step in dynamical systems simulation. ode-toolbox assists in solver benchmarking, and recommends a solver on the basis of a set of user-configurable heuristics. If some or all of the dynamical equations are found to be analytically tractable, ode-toolbox generates corresponding propagator matrices, that allow a solution to be calculated at machine precision.

ode-toolbox is written in Python and leverages SymPy for the symbolic manipulation of equations. It was initially developed in the context of the [NESTML](https://github.com/nest/nestml) project, in which the main focus was on the class of spiking neurons presently available in the [NEST](https://github.com/nest/nest-simulator) simulator. It can, however, be used standalone and is broadly applicable to continuous-time dynamical systems as well as systems that undergo instantaneous events (such as neuronal spikes or impulsive forces).


## Installation

### Prerequisites

ode-toolbox depends on the Python packages SymPy, SciPy and NumPy and (optionally) matplotlib and graphviz for visualisation, and pytest for self-tests. The stiffness tester additionally depends on an installation of [PyGSL](http://pygsl.sourceforge.net/). If PyGSL is not installed, the test for stiffness is skipped during the analysis of the equations.

The required packages can be installed by running 

```
pip install sympy scipy numpy pygsl
```


### Installing ode-toolbox

To install, clone the repository, go to the root directory and then run the following commands in a terminal:

```
python setup.py install
```

If you wish to install ode-toolbox into your home directory, add the option `--user` to the above call.

For further installation hints, please see [.travis.yml](.travis.yml).


### Testing

To run the unit and integration tests that come with ode-toolbox, you can run the following command:

```
python setup.py test
```

Please note that this requires the [pytest](https://docs.pytest.org) package to be installed.


## Usage

The `ode-toolbox` can be used in two ways:
1. as a Python module. See [the tests](tests/test_ode_analyzer.py) for examples of the exact usage of the functions.
2. as command line application. In this case, the input is stored in a `json` file, whose file format will be explained in the next section. The command line invocation in this case looks like this:
```
ode_analyzer.py <json_file>
```


## Solver selection criteria

TODO


## Input

The JSON input dictionary that is passed to ode-toolbox contains **dynamics**, **numerical parameters**, and **options**. **Documentation** may optionally be provided as a string.

All expressions are parsed as sympy expressions. There are several predefined symbols, such as `e` and `E` for Euler's number, trigonometric functions, etc. The list of predefined symbols is defined in `symbols.py`, as the static member `Shape._sympy_globals`. Variable names should be chosen such that they do not overlap with the predefined symbols.


### Dynamics

All dynamical variables have a variable name, a differential order, and a defining expression. The overall dynamics is given as a list of these definitions. For example, we can define an alpha shape kernel :math:`g` with time constant :math:`\tau` as follows:

```Python
"dynamics":
[
    {
        "expression": "g'' = -1 / tau**2 * g - 2/tau * g'"
    }
]
```

Instead of a second-order differential equation, we can equivalently describe the kernel shape as a function of time:

```Python
"dynamics":
[
    {
        "expression": "g = (e / tau) * t * exp(-t / tau)"
    }
]
```

Expressions can refer to variables defined in other expressions. For example, a third equivalent formulation of the alpha shape is as the following system of two coupled first-order equations:

```Python
"dynamics":
[
    {
        "expression": "g' = h",
        "initial_value": "0"
    },
    {
        "expression": "h' = -g / tau**2 - 2 * h / tau",
        "initial_value": "e / tau"
    }
]
```


### Initial values

As many initial values have to be specified as the differential order requires, that is, none for functions of time, one for a one-dimensional system of ODEs, and so on. Continuing the second-order alpha function example:

```Python
"dynamics":
[
    {
        "expression": "g = (e / tau) * t * exp(-t / tau)"
        "initial_values":
        {
            "g" : "0",
            "g'" : "e / tau"
        }
    }
]
```

If only one initial value is required, the following simpler syntax may be used, which omits the variable name:

```Python
"dynamics":
[
    {
        "expression": "g' = -g / tau"
        "initial_value": "e / tau"
    }
]
```

### Parameters

It is not necessary to supply numerical values for the parameters (e.g. `tau = 10E-3`). The expressions are symbolically analysed, and in this case a set of propagators will be generated, because the alpha kernel is analytically tractable. However, in some cases (in particular when doing stiffness testing), it can be important to simulate with a particular set of parameter values. In this case, they can be specified in the global `parameters` dictionary. This dictionary maps parameter names to default values, for example:


```Python
"parameters":
{
    "C_m": ".4",
    "E_K": "-70",
    "E_L": "-50",
    "I_ext": "30E-3"
}
```

### Options

Further options for the integrator, decision criteria for solver selection and so on, can be specified in the global `options` dictionary, for example:

```Python
"options" : {
    "output_timestep_symbol": "__h"
    "sim_time": "100E-3",
    "max_step_size": ".25E-3",
    "integration_accuracy" : "1E-9"
}
```


### Upper and lower thresholds

TODO


## Output

The analysis output is returned in the form of a Python dictionary, or an equivalent JSON file.

TODO

