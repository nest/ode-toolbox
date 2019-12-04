# ode-toolbox

[![Build status](https://travis-ci.org/nest/ode-toolbox.svg?branch=master)](https://travis-ci.org/nest/ode-toolbox) [![Testing coverage](https://codecov.io/gh/nest/ode-toolbox/branch/master/graph/badge.svg)](https://codecov.io/gh/nest/ode-toolbox)

Choosing the optimal solver for systems of ordinary differential equations (ODEs) is a critical step in dynamical systems simulation. ode-toolbox assists in solver benchmarking, and recommends solvers on the basis of a set of user-configurable heuristics. For all dynamical equations that admit an analytic solution, ode-toolbox generates propagator matrices that allow the solution to be calculated at machine precision.

<p align="center">
<img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/flow_diagram.png" alt="Flow diagram" width="361" height="341">
</p>

ode-toolbox is written in Python and leverages SymPy for the symbolic manipulation of equations. It was initially developed in the context of the [NESTML](https://github.com/nest/nestml) project, in which the main focus was on the class of spiking neurons presently available in the [NEST](https://github.com/nest/nest-simulator) simulator. It can, however, be used standalone and is broadly applicable to continuous-time dynamical systems as well as systems that undergo instantaneous events (such as neuronal spikes or impulsive forces).


## Installation

### Prerequisites

Only Python 3 is supported. ode-toolbox depends on the Python packages SymPy, SciPy and NumPy and (optionally) matplotlib and graphviz for visualisation, and pytest for self-tests. The stiffness tester additionally depends on an installation of [PyGSL](http://pygsl.sourceforge.net/). If PyGSL is not installed, the test for stiffness is skipped during the analysis of the equations.

All required and optional packages can be installed by running 

```
pip install -r requirements.txt
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
python -m pytest
```

Please note that this requires the [pytest](https://docs.pytest.org) package to be installed.

To increase the verbosity, append the command-line parameters `-s -o log_cli=true -o log_cli_level="DEBUG"`.


## Usage

ode-toolbox can be used in two ways:
1. As a Python module. Import the `odetoolbox` module, and then call `odetoolbox.analysis(indict)` where `indict` is the JSON-like input in Python dictionary format. See the tests (e.g. [test_lorenz_attractor.py](tests/test_lorenz_attractor.py)) for an example.
2. As command line application. In this case, the input is stored in a JSON file, and ode-toolbox is invoked from the command line as <code>ode_analyzer.py [lorenz_attractor.json](tests/lorenz_attractor.json)</code>

The JSON file and Python dictionary are completely equivalent in content and form, described in the "Input" section below.

## Input

The JSON input dictionary that is passed to ode-toolbox contains **dynamics**, **numerical parameters**, and **global options**. **Documentation** may optionally be provided as a string.

All expressions are parsed as sympy expressions, and subsequently simplified through `sympy.simplify()`. There are several predefined symbols, such as `e` and `E` for Euler's number, trigonometric functions, etc. `t` is assumed to represent time. The list of predefined symbols is defined in `symbols.py`, as the static member `Shape._sympy_globals`. Variable names should be chosen such that they do not overlap with the predefined symbols.


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
    },
    {
        "expression": "h' = -g / tau**2 - 2 * h / tau",
    }
]
```


### Initial values

As many initial values have to be specified as the differential order requires, that is, none for functions of time, one for a one-dimensional system of ODEs, and so on. Continuing the second-order alpha function example:

```Python
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

### Upper and lower thresholds

Neuronal dynamics is typically characterised by a discontinuous jump upon action potential firing. To model this behaviour, an upper and lower bound can be defined for each input variable. When either bound is reached, the state of that variable is reset to its initial value.

Thresholds are mainly of interest when doing stiffness testing, and only apply to equations that are solved by the numerical integrator. Testing for threshold crossing and reset of the state variable(s) occurs at the beginning of every timestep.

```Python
"dynamics":
[
    {
      "expression": "V_m' = (-g_L * (V_m - E_L) - g_ex * (V_m - E_ex)) / C_m
      "initial_value": "-70",
      "upper_bound": "-55"
    }
}
```


### Parameters

It is not necessary to supply any numerical values for parameters. The expressions are symbolically analysed, and in some cases a set of symbolic propagators will be generated. However, in some cases (in particular when doing stiffness testing), it can be important to simulate with a particular set of parameter values. In this case, they can be specified in the global `parameters` dictionary. This dictionary maps parameter names to default values, for example:

```Python
"parameters":
{
    "N": "10",
    "C_m": "400.",
    "tau": "1 - 1/e",
    "I_ext": "30E-3"
}
```


## Spiking stimulus for stiffness testing

Spike times for each variable can be read directly from the JSON input as a list, or be generated according to a constant frequency or Poisson distribution. The general format is as follows: any number of stimuli can be defined in the global list `"stimuli"`. Each entry in the list is a dictionary containing parameters, and a `"variables"` attribute that specifies which dynamical variables are affected by this particular spike generator. For example:

```Python
"stimuli":
[
    {
        "type": "poisson_generator",
        "rate": "10.",
        "variables": ["g_in'", "g_ex'"]
    }
]
```

The type is one of `"poisson_generator"`, `"regular"` or `"list"`. The Poisson and regular spiking generators only have one parameter: rate. When the selected type is `"list"`, a list of predefined spike times can be directly supplied under the key `"list"`, separated by spaces, as such:

```Python
{
    "type": "list",
    "list": "5E-3 10E-3 20E-3 15E-3 50E-3",
    "variables": ["I'"]
}
```


### Global options

Further options for the integrator, decision criteria for solver selection and so on, can be specified in the global `options` dictionary, for example:

```Python
"options" : {
    "sim_time": "100E-3",
    "max_step_size": ".25E-3"
}
```

The following global options are defined. Note that all are typically formatted as strings when encoding into JSON.

| Name | Type | Default | Description  |
| ------------- | ------------- | ------------- | ----- |
| `integration_accuracy_abs` | 1E-9 | float | Absolute error bound for all numerical integrators that are used. |
| `integration_accuracy_rel` | 1E-9 | float | Relative error bound for all numerical integrators that are used. |
| `output_timestep_symbol` | `"__h"` | string | Generated propagators are a function of the simulation timestep. This parameter gives the name of the variable that contains the numerical value of the timestep during simulation. |
| `sim_time` | 100E-3 | float | Total simulated time. |
| `max_step_size` | 999 | float | Maximum step size during simulation (e.g. for stiffness testing solvers). |
| `differential_order_symbol` | `"__d"` | string | String appended n times to output variable names to indicate differential order n. XXX: TODO: only the default value works for now. |


## Output

The analysis output is returned in the form of a Python dictionary, or an equivalent JSON file.

During analysis, ode-toolbox rewrites the differential notation from single quotation marks into characters that are typically compatible with variable names; by default every quotation mark is rewritten into the string specified as the global parameter `differential_order_symbol` (by default, `"__d"`).

ode-toolbox will return a list of solvers. Each solver has the following keys:
- `solver`: a string containing the solver recommendation. Starts with either "analytical" or "numeric".
- `state_variables`: an unordered list containing all variable symbols.
- `initial_values`: a dictionary that maps each variable symbol (in string form) to a sympy expression. For example `"g" : "e / tau"`.
- `parameters`: only present when parameters were supplied in the input. The input parameters are copied into the output for convenience.

Analytic solvers have the following extra entries:

- `update_expressions` : a dictionary that maps each variable symbol (in string form) to a sympy propagator expression. The interpretation of an entry `"g" : "g * __P__g__g + h * __P__g__h"` is that, at each integration timestep, when the state of the system needs to be updated from the current time :math:`t` to the next step :math:`t + \Delta t`, we assign the new value `"g * __P__g__g + h * __P__g__h"` to the variable `g`. Note that the expression is always evaluated at the old time :math:`t`; this means that when more than one state variable needs to be updated, all of the expressions have to be calculated before updating any of the variables.
- `propagators` : a dictionary that maps each propagator matrix entry to its defining expression; for example `"__P__g__h" : "__h*exp(-__h/tau)"`


Numeric solvers have the following extra entries:
- `update_expressions`: a dictionary that maps each variable symbol (in string form) to a sympy expression that is its Jacobian, that is, for a symbol :math:`x`, the expression is equal to :math:`\frac{\delta x}{\delta t}`.


## Analytic solver selection criteria

If an ODE is homogeneous, constant-coefficient and linear, an analytic solution can be computed. Analytically solvable ODEs can also contain dependencies on other analyically solvable ODEs, but an otherwise analytically tractable ODE cannot depend on an ODE that can only be solved numerically. In the latter case, no analytic solution will be computed.

For example, consider an integrate-and-fire neuron with two alpha-shaped kernels (`I_shape_in` and `I_shape_gap`), and one nonlinear kernel (`I_shape_ex`). Each of these kernels can be expressed as a system of ODEs containing two variables. `I_shape_in` is specified as a second-order equation, whereas `I_shape_gap` is explicitly given as a system of two coupled first-order equations, i.e. as two separate `dynamics` entries with names `I_shape_gap1` and `I_shape_gap2`.

Both formulations are mathematically equivalent, and ode-toolbox treats them the same following input processing.

During processing, a dependency graph is generated, where each node corresponds to one dynamical variable, and an arrow from node *a* to *b* indicates that *a* depends on the value of *b*. Boxes enclosing nodes mark input shapes that were specified as either a direct function of time or a higher-order differential equation, and were expanded to a system of first-order ODEs.

<p align="center">
<img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/eq_analysis_0.png" alt="Dependency graph" width="620" height="283">
</p>

Each variable is subsequently marked according to whether it can, by itself, be analytically solved. This is indicated by a green colour.

<p align="center">
<img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/eq_analysis_1.png" alt="Dependency graph with membrane potential and excitatory and gap junction kernels marked green" width="720" height="383">
</p>

Second, variables are unmarked as analytically solvable if they depend on other variables that are themselves not analytically solvable. In this example, `V_abs` is unmarked as it depends on the nonlinear excitatory kernel.

<p align="center">
<img src="https://raw.githubusercontent.com/clinssen/ode-toolbox/merge_shape_ode_concepts-dev/doc/fig/eq_analysis_2.png" alt="Dependency graph with membrane potential and excitatory and gap junction kernels marked green" width="720" height="383">
</p>

The analytic solution for all green nodes is computed in the form of a propagator matrix. See the section "Analytic solver generation" for more details.


## Numeric solver selection criteria

Solver selection is performed on the basis of a set of rules, defined in `StiffnessTester.draw_decision()`. The logic is as follows:

 * If the minimum step size recommended by all solvers is smaller than `machine_precision_dist_ratio` times the machine precision, a warning is issued.
 * If the minimum step size for the implicit solver is smaller than `machine_precision_dist_ratio` times the machine precision, recommend the explicit solver.
 * If the minimum step size for the explicit solver is smaller than `machine_precision_dist_ratio` times the machine precision, recommend the implicit solver.
 * If the average step size for the implicit solver is at least `avg_step_size_ratio` times as large as the average step size for the explicit solver, recommend the implicit solver.
 * Otherwise, recommend the explicit solver.


| Name        | Default           | Description  |
| ------------- | ------------- | ----- |
| `avg_step_size_ratio` | 6 | Ratio between average step sizes of implicit and explicit solver. Larger means that the explicit solver is more likely to be selected. |
| `machine_precision_dist_ratio` | 10 | Disqualify a solver if its minimum step size comes closer than this ratio to the machine precision. |




## Internal representation

For users who want to modify/extend ode-toolbox

Processing input dynamics: method from Blundell et al. 2018

The aim is to find a representation of the form `a0 * f + a1 * f' + ... + an * f^(n) = 0`.

* For direct functions of time `f(t)`:
  1. Find `t` such that `f(t) ≠ 0`
  2. Test if `f` can be expressed in the first-order form `f' = a_0 * f`
  3. If not: test if `f` can be expressed in n-th order form ...

Internal representation as `SystemOfShapes`, generation of matrices A and C such that x' = Ax + C


## Analytic solver generation

The propagator matrix `P` is derived from the system matrix by matrix exponentiation:

`P = exp(A · h)`

If the imaginary unit *i* is found in any of the entries in `P`, fail. This usually indicates an unstable (diverging) dynamical system. Double-check the dynamical equations.

In some cases, elements of `P` may contain fractions that have a factor of the form `param1 - param2` in their denominator. If at a later stage, the numerical value of `param1` is chosen equal to that of `param2`, a numerical singularity (division by zero) occurs. To avoid this issue, it is necessary to eliminate either `param1` or `param2` in the input, before the propagator matrix is generated.


## Working with large expressions

Performance issues with sympy; `Shape.EXPRESSION_SIMPLIFICATION_THRESHOLD`

A caching mechanism will be implemented in the future.


## Contributions and getting help

The primary development of ode-toolbox happens on GitHub, at https://github.com/nest/ode-toolbox. If you encounter any issue, please create an new entry in the GitHub issue tracker. Pull requests are welcome.


## Citations

1. Inga Blundell, Dimitri Plotnikov, Jochen Martin Eppler and Abigail Morrison (2018) **Automatically selecting a suitable integration scheme for systems of differential equations in neuron models.** Front. Neuroinform. [doi:10.3389/fninf.2018.00050](https://doi.org/10.3389/fninf.2018.00050). Preprint available on [Zenodo](https://zenodo.org/record/1411417).


## Acknowledgments

This software was initially supported by the JARA-HPC Seed Fund *NESTML - A modeling language for spiking neuron and synapse models for NEST* and the Initiative and Networking Fund of the Helmholtz Association and the Helmholtz Portfolio Theme *Simulation and Modeling for the Human Brain*.

This software was developed in part or in whole in the Human Brain Project, funded from the European Union's Horizon 2020 Framework Programme for Research and Innovation under Specific Grant Agreements No. 720270 and No. 785907 (Human Brain Project SGA1 and SGA2).
