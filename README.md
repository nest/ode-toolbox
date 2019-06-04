# NEST ODE-toolbox

[![Build Status](https://travis-ci.org/nest/ode-toolbox.svg?branch=master)](https://travis-ci.org/nest/ode-toolbox)

In the computer simulation of dynamical systems, choosing the most suitable differential equation solver might depend on many factors. ODE-toolbox assists in picking the optimal solver for a given set of dynamical equations (also known as "ordinary differential equations" or "initial value problems"). ODE-toolbox will generate propagator matrices for the subset of dynamics that is linear and constant coefficient, which can be used to solve the equations up to machine precision. For those equations that are not analytically tractable, an optimal numerical integrator (e.g. explicit vs. implicit) is recommended based on several user-configurable heuristics.

ODE-toolbox is written in Python and relies on SymPy for the symbolic manipulation of equations. It was initially developed in the context of the [NESTML](https://github.com/nest/nestml) project, in which the main focus was on the class of spiking neurons presently available in the [NEST](https://github.com/nest/nest-simulator) simulator. It can, however, be used standalone and is broadly applicable to continuous-time dynamical systems as well as systems that undergo instantaneous events (such as neuronal spikes or impulsive forces).


## Installation

### Prerequisites

ODE-toolbox depends on the Python packages SymPy, SciPy and NumPy and (optionally) matplotlib and graphviz for visualisation, and pytest for self-tests. The stiffness tester additionally depends on an an installation of [PyGSL](http://pygsl.sourceforge.net/). If PyGSL is not installed, the test for stiffness is skipped during the analysis of the equations.

The required packages can be installed by running 

```
pip install sympy scipy numpy
```


### Installing ODE-toolbox

To install, run the following commands in a terminal:

```
python setup.py install
```

If you wish to install ODE-toolbox into your home directory, add the option `--user` to the above call.

For further installation hints, please see [.travis.yml](.travis.yml).


### Testing

To run the unit and integration tests that come with ODE-toolbox, you can run the following command:

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

### Input

Input for `ode_analyzer.py` JSON files are composed of two lists of dictionaries `shapes` and `odes` and a dictionary `parameters`. `shapes` and `odes` must be stated at any call of the `ode_analyzer.py`. `parameters` dictionary is stated only if a stiffness test should be performed.

Note that there are several predefined symbols, such as `e` and `E` for Euler's number, trigonometric functions, etc. The list of predefined symbols is defined in `symbols.py`, as the static member `Shape._sympy_globals`. Variable names should be chosen such that they do not overlap with the predefined symbols.


#### `odes`

The ODEs entry contains dictionaries, each of which specifies an ODE with initial values. Each of these dictionaries has the following keys:

* `symbol`: The unambiguous variable name that this ODE describes the dynamics of
* `definition`: An arbitrary Python expression with free variables (an arbitrary valid Python variable name, e.g. `V_m`), derivative-variables (an arbitrary valid Python variable name with a postfix of a sequence of '-characters, e.g. `g_in'''`) and functions from the `math`-package. The definition of a `function` must depend on `t`-variable.
* `upper_bound`: defines an optional condition as a valid boolean Python expression for the maximal value of the `symbol`-variable. This bound is used in the stiffness test.
* `lower_bound`: defines an optional condition as a valid boolean Python expression for the minimal value of `symbol`-variable. This bound is used in the stiffness test.
* `initial_values`: A dictionary containing `order` many key-value pairs: one for each derivative that occurs in the ODE. The keys are strings created by concatenating the variable symbol with as many single quotation marks (') as the derivation order. The values are SymPy expressions.

#### `parameters`

Model parameters and their values are given in the `parameters` dictionary. This dictionary maps default values to parameter names and has to contain an entry for each free variable occurring in the equations of the `odes` or `shapes`. Every dictionary entry hat the name of the free variable as its key and a valid Python expression as its value. I.e. `variable_name: expression`.

#### `shapes`

Shapes-list contains shapes which can be selectively specified as a function of time (by referencing a predefined variable `t`) or as an ODE with initial conditions. Every shape is a dictionary with the following keys:

* `type`: determines whether the shape is given as a function of time (`function`) or as an ODE with initial conditions (`ode`)
* `symbol`: The unambiguous name of the shape
* `definition`: An arbitrary Python expression with free variables (an arbitrary valid Python variable name, e.g. `V_m`), derivative-variables (an arbitrary valid Python variable name with a postfix of a sequence of '-characters, e.g. `g_in'''`) and functions from the `math`-package. The definition of a `function` must depend on `t`-variable.
* `initial_values`: Initial values of the ODE representing the shape. The dictionary contains `order` many key-value pairs: one for each derivative that occurs in the ODE. The keys are strings created by concatenating the variable symbol with as many single quotation marks (') as the derivation order. The values are SymPy expressions. Must be stated only for the `ode`-shape.

#### Output

The analysis output is returned in the form of a Python dictionary. If the input was given as JSON in a file `input.json`, the output is stored in a file `input_result.json` in the current working directory.

* `solver`: states which integration scheme was selected. Possible values are `analytical` or `numerical`.
  * if the stiffness tester was run (i.e. if PyGSL is available), this field also contains either `explicit` of `implicit` after a dash (`-`).
* `ode_updates`: dictionary mapping ODE variable symbols to dictionary containing one (key, value)-pair for each term that contributes to the dynamics: a constant term (with key `constant`), and one more pair for each shape, with key equal to the variable symbol of that shape.
* `shape_initial_values`: are the initial values of the differentialequations the shapes satisfy.
* `shape_state_updates`: contains a dictionary that contains, for each differential order of the variable symbol, instructions to be applied to in every update step if the ODE is solved analytically.
* `propagator`: contains a list of all non-zero entries of all propagator matrices.
* `ode_system_type`: states if the current ode system with provided parameters is a stiff and non-stiff problem. Possible values `stiff`, `non-stiff`, `skipped` (if due to missing `PyGSL` or `parameters` the test was not performed)

## Examples

### Analytically solvable current-based model

The following example shows an input model that corresponds to a current-based integrate-and-fire neuron with an alpha-shaped postsynaptic response. The model ODEs can be solved analytically.

```Python
{
  "shapes": [
    {
      "type": "function",
      "symbol": "I_shape_in",
      "definition": "(e/tau_syn_in) * t * exp(-t/tau_syn_in)"
    },
    {
      "type": "ode",
      "symbol": "I_shape_ex",
      "definition": "(-1)/(tau_syn_ex)**(2)*I_shape_ex+(-2)/tau_syn_ex*I_shape_ex'",
      "initial_values": {
        "I_shape_ex" : "0",
        "I_shape_ex'" : "e / tau_syn_ex"
      }
    }
  ],

  "odes": [
    {
      "symbol": "V_abs",
      "definition": "(-1)/Tau*V_abs+1/C_m*(I_shape_in+I_shape_ex+I_e+currents)"
    }
  ]

}

```

Output:

```Python
{
  "solver": "analytical",
  "ode_updates": {
    "V_abs": {
      "I_shape_in": "I_shape_in*__P_I_shape_in__2_1 + I_shape_in__d*__P_I_shape_in__2_0",
      "I_shape_ex": "I_shape_ex*__P_I_shape_ex__2_1 + I_shape_ex__d*__P_I_shape_ex__2_0",
      "constant": "(exp(-__h/Tau)) * V_abs + ((I_e + currents)/C_m) * (Tau - Tau*exp(-__h/Tau))"
    }
  },
  "shape_initial_values": {
    "I_shape_in": {
      "I_shape_in": "0",
      "I_shape_in'": "e/tau_syn_in"
    },
    "I_shape_ex": {
      "I_shape_ex": "0"
      "I_shape_ex'": "e/tau_syn_ex",
    }
  },
  "shape_state_updates": {
    "I_shape_in": {
      "I_shape_in": "I_shape_in*__P_I_shape_in__1_1 + I_shape_in__d*__P_I_shape_in__1_0",
      "I_shape_in'": "I_shape_in*__P_I_shape_in__0_1 + I_shape_in__d*__P_I_shape_in__0_0"
    },
    "I_shape_ex": {
      "I_shape_ex": "I_shape_ex*__P_I_shape_ex__1_1 + I_shape_ex__d*__P_I_shape_ex__1_0"
      "I_shape_ex'": "I_shape_ex*__P_I_shape_ex__0_1 + I_shape_ex__d*__P_I_shape_ex__0_0",
    }
  },
  "propagator": {
    "__P_I_shape_in__0_0": "(-__h + tau_syn_in)*exp(-__h/tau_syn_in)/tau_syn_in"
    "__P_I_shape_in__0_1": "-__h*exp(-__h/tau_syn_in)/tau_syn_in**2",
    "__P_I_shape_in__1_0": "__h*exp(-__h/tau_syn_in)",
    "__P_I_shape_in__1_1": "(__h + tau_syn_in)*exp(-__h/tau_syn_in)/tau_syn_in",
    "__P_I_shape_in__2_0": "Tau*tau_syn_in*(Tau*tau_syn_in*exp(__h/tau_syn_in) - (Tau*tau_syn_in + __h*(Tau - tau_syn_in))*exp(__h/Tau))*exp(-__h*(Tau + tau_syn_in)/(Tau*tau_syn_in))/(C_m*(Tau**2 + Tau*(-2*Tau + tau_syn_in) + (Tau - tau_syn_in)*(2*Tau - tau_syn_in)))",
    "__P_I_shape_in__2_1": "-Tau*(Tau*__h*exp(__h/Tau) + 2*Tau*tau_syn_in*exp(__h/Tau) - 2*Tau*tau_syn_in*exp(__h/tau_syn_in) - __h*tau_syn_in*exp(__h/Tau) - tau_syn_in**2*exp(__h/Tau) + tau_syn_in**2*exp(__h/tau_syn_in))*exp(-__h/tau_syn_in - __h/Tau)/(C_m*(Tau**2 - 2*Tau*tau_syn_in + tau_syn_in**2))",
    "__P_I_shape_in__2_2": "exp(-__h/Tau)",
    "__P_I_shape_ex__0_0": "(-__h + tau_syn_ex)*exp(-__h/tau_syn_ex)/tau_syn_ex",
    "__P_I_shape_ex__0_1": "-__h*exp(-__h/tau_syn_ex)/tau_syn_ex**2",
    "__P_I_shape_ex__1_0": "__h*exp(-__h/tau_syn_ex)",
    "__P_I_shape_ex__1_1": "(__h + tau_syn_ex)*exp(-__h/tau_syn_ex)/tau_syn_ex",
    "__P_I_shape_ex__2_0": "Tau*tau_syn_ex*(Tau*tau_syn_ex*exp(__h/tau_syn_ex) - (Tau*tau_syn_ex + __h*(Tau - tau_syn_ex))*exp(__h/Tau))*exp(-__h*(Tau + tau_syn_ex)/(Tau*tau_syn_ex))/(C_m*(Tau**2 + Tau*(-2*Tau + tau_syn_ex) + (Tau - tau_syn_ex)*(2*Tau - tau_syn_ex)))",
    "__P_I_shape_ex__2_1": "-Tau*(Tau*__h*exp(__h/Tau) + 2*Tau*tau_syn_ex*exp(__h/Tau) - 2*Tau*tau_syn_ex*exp(__h/tau_syn_ex) - __h*tau_syn_ex*exp(__h/Tau) - tau_syn_ex**2*exp(__h/Tau) + tau_syn_ex**2*exp(__h/tau_syn_ex))*exp(-__h/tau_syn_ex - __h/Tau)/(C_m*(Tau**2 - 2*Tau*tau_syn_ex + tau_syn_ex**2))",
    "__P_I_shape_ex__2_2": "exp(-__h/Tau)",
  }
}
```

### Conductance-based model

The following example shows an input model that corresponds to a conductance-based integrate-and-fire neuron with an alpha-shaped postsynaptic response. This example also provides a `parameters` dictionary for stiffness testing.

```Python
{
  "shapes": [
    {
      "type": "function",
      "symbol": "g_in",
      "definition": "(e/tau_syn_in)*t*exp((-1)/tau_syn_in*t)"
    },
    {
      "type": "ode",
      "symbol": "g_ex",
      "definition": "(-1)/(tau_syn_ex)**(2)*g_ex+(-2)/tau_syn_ex*g_ex'",
      "initial_values": {
        "g_ex" : "0",
        "g_ex'" : "e / tau_syn_ex"
      }
    }
  ],

  "odes": [
    {
      "symbol": "V_m",
      "definition": "(-(g_L*(V_m-E_L))-(g_ex*(V_m-E_ex))-(g_in*(V_m-E_in))+I_stim+I_e)/C_m",
      "initial_values": {"V_m" : "E_L"},
      "upper_bound": "V_th"
    }
  ],

  "parameters": {
    "V_th": "-55.0",
    "g_L": "16.6667",
    "C_m": "250.0",
    "E_ex": "0",
    "E_in": "-85.0",
    "E_L": "-70.0",
    "tau_syn_ex": "0.02",
    "tau_syn_in": "2.0",
    "I_e": "0",
    "I_stim": "0"
  }
}
```

Output:

```Python
{
  "solver": "numeric-implicit",
  "shape_initial_values": {
    "g_ex": {
      "g_ex": "0",
      "g_ex'": "e/tau_syn_ex"
    },
    "g_in": {
      "g_in": "0",
      "g_in'": "e/tau_syn_in"
    }
  },
  "shape_ode_definitions": {
    "g_ex": "-1/tau_syn_ex**2 * g_ex + -2/tau_syn_ex * g_ex__d",
    "g_in": "-1/tau_syn_in**2 * g_in + -2/tau_syn_in * g_in__d"
  }
}
```
