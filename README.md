# ODE Toolbox - a framework for automatic solver selection for systems of differential equations 

## Prerequisites 
* ode-toolbox requires `sympy` in the version of at least `1.1.1` (`pip install sympy` installs a satisfying version)
* For the stiffness testing, an installation of `PyGSL` is necessary (http://pygsl.sourceforge.net/). If `PyGSL` is not installed then the stiffness testing is omitted but the remainder of the analysis framework is still working.

## Installation

```Python
python setup.py install
```

### Testing
```Python
python setup.py test
```

## Usage of the analysis framework
The `ode-toolbox` can be used either as a normal python module (cf.  tests in `tests` to see the API) or as command line application. The input for `ode-toolbox` is stored in `json`-files (file format will be explained in the next section). The entry point for the analysis is `ode_analyzer.py`, The script expects the name of a JSON file as its only command line argument:

```
python ode_analyzer.py iaf_cond_alpha.json
```
`ode_analyzer.py` stores output in an JSON-file named `result-$date$.json` where `$date$` correspond to the current date.
## Input
Input for `ode_analyzer.py` JSON files are composed of two lists of dictionaries `shapes` and `odes` and a dictionary `parameters`. `shapes` and `odes` must be stated at any call of the `ode_analyzer.py`. `parameters` dictionary is stated only if a stiffness test should be performed.
### shapes
Shapes-list contains shapes which can be selectively specified as a function of time (by referencing a predefined variable `t`) or as an ODE with initial conditions. Every shape is a dictionary with the following keys:

* `type`: determines whether the current shape is a function (`function`) or an ODE with initial conditions (`ode`)
* `symbol`:  The unambiguous name of the shape
* `definition`: An arbitrary Python-expression with free variables (an arbitrary valid Python-variable name, e.g. `V_m`), derivative-variables (an arbitrary valid Python-variable name with a postfix of a sequence of '-characters, e.g. `g_in'''`) and functions from the `math`-package. The definition of a `function` must depend on `t`-variable.
* `initial_values`: A list with Python-expressions which define an initial value for every order of the shape-ODE. The length of this list defines the order of the corresponding ODE. `initial_values` must be stated only for the `ode`-shape.

### odes
Odes-list contains  dictionaries each of which specifies an ODE with initial values. Every dictionary has the following keys:

* `symbol`:  The unambiguous name of the shape
* `definition`: An arbitrary Python-expression with free variables (an arbitrary valid Python-variable name, e.g. `V_m`), derivative-variables (an arbitrary valid Python-variable name with a postfix of a sequence of '-characters, e.g. `g_in'''`) and functions from the `math`-package. The definition of a `function` must depend on `t`-variable.
* `upper_bound`: defines an optional condition as a valid boolean Python-expression for the maximal value of the `symbol`-variable. This bound is used in the stiffness test.
* `lower_bound`: defines an optional condition as a valid boolean Python-expression for the minimal value of `symbol`-variable. This bound is used in the stiffness test.
* `initial_values`: A list with Python-expressions which define an initial value for every order of the shape-ODE. The length of this list defines the order of the corresponding ODE. `initial_values` must be stated only for the `ode`-shape.


### parameters
Model parameters and their values are given in the `parameters` dictionary. This dictionary maps default values to parameter names and has to contain an entry for each free variable occurring in the equations of the `odes` or `shapes`. Every dictionary entry hat the name of the free variable as its key and a valid Python-expression as its value. I.e. `variable_name: expression`.

## Output
The analysis output is stored as JSON in file `result-$date$.json` where `$date$` correspond to the current date.

* `solver`: states which integration scheme was selected. Possible values are `analytical` or `numerical `.
* `ode_updates`: contains a list of Python statements for updating the ODE.
* `state_shape_variables`: is a list of the names of the shapes and their derivatives up to the order of the ODE they satisfy minus one.
* `shape_initial_values`: are the initial values of the differential equations the shapes satisfy.
* `shape_state_updates`: contains a list of instructions to be applied to shape_state_variables in every update step if the ODE is solved analytically.
* `propagator`: contains a list of all non-zero entries of all propagator matrices numbered according to the order in which they should be executed if the ODE is solved analytically.
* `ode_system_type`: states if the current ode system with provided parameters is a stiff and non-stiff problem. Possible values `stiff`, `non-stiff`, `skipped` (if due to missing `PyGSL` or `parameters` the test was not performed)

## Examples
### Analytically solvable current-based model
The following example shows an input model that corresponds to a current-based integrate-and-fire neuron with an alpha-shaped postsynaptic response. The model ODEs can be solved analytically.
```
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
      "initial_values": ["0",  "e / tau_syn_ex"]
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
```
{
  "ode_updates": [
    "V_abs = (exp(-__h/Tau)) * V_abs+ ((I_e + currents)/C_m) * (Tau - Tau*exp(-__h/Tau))", 
    "V_abs += I_shape_in*__P_I_shape_in__2_1 + I_shape_in__d*__P_I_shape_in__2_0", 
    "V_abs += I_shape_ex*__P_I_shape_ex__2_1 + I_shape_ex__d*__P_I_shape_ex__2_0"
  ], 
  "solver": "analytical", 
  "shape_initial_values": [
    [
      "0", 
      "e/tau_syn_in"
    ], 
    [
      "0", 
      "e/tau_syn_ex"
    ]
  ], 
  "shape_state_updates": [
    [
      "I_shape_in__d*__P_I_shape_in__0_0", 
      "I_shape_in*__P_I_shape_in__1_1 + I_shape_in__d*__P_I_shape_in__1_0"
    ], 
    [
      "I_shape_ex__d*__P_I_shape_ex__0_0", 
      "I_shape_ex*__P_I_shape_ex__1_1 + I_shape_ex__d*__P_I_shape_ex__1_0"
    ]
  ], 
  "shape_state_variables": [
    [
      "I_shape_in__d", 
      "I_shape_in"
    ], 
    [
      "I_shape_ex__d", 
      "I_shape_ex"
    ]
  ], 
  "propagator": {
    "__P_I_shape_in__1_1": "exp(-__h/tau_syn_in)", 
    "__P_I_shape_in__1_0": "__h*exp(-__h/tau_syn_in)", 
    "__P_I_shape_in__0_0": "exp(-__h/tau_syn_in)", 
    "__P_I_shape_ex__2_2": "exp(-__h/Tau)", 
    "__P_I_shape_ex__2_0": "Tau*tau_syn_ex*(Tau*tau_syn_ex*exp(2*__h/tau_syn_ex) - Tau*tau_syn_ex*exp(__h*(Tau + tau_syn_ex)/(Tau*tau_syn_ex)) - __h*(Tau - tau_syn_ex)*exp(__h*(Tau + tau_syn_ex)/(Tau*tau_syn_ex)))*exp(-__h*(2*Tau + tau_syn_ex)/(Tau*tau_syn_ex))/(C_m*(Tau - tau_syn_ex)**2)", 
    "__P_I_shape_ex__2_1": "Tau*tau_syn_ex*(-exp(__h/Tau) + exp(__h/tau_syn_ex))*exp(-__h*(Tau + tau_syn_ex)/(Tau*tau_syn_ex))/(C_m*(Tau - tau_syn_ex))", 
    "__P_I_shape_in__2_0": "Tau*tau_syn_in*(Tau*tau_syn_in*exp(2*__h/tau_syn_in) - Tau*tau_syn_in*exp(__h*(Tau + tau_syn_in)/(Tau*tau_syn_in)) - __h*(Tau - tau_syn_in)*exp(__h*(Tau + tau_syn_in)/(Tau*tau_syn_in)))*exp(-__h*(2*Tau + tau_syn_in)/(Tau*tau_syn_in))/(C_m*(Tau - tau_syn_in)**2)", 
    "__P_I_shape_in__2_1": "Tau*tau_syn_in*(-exp(__h/Tau) + exp(__h/tau_syn_in))*exp(-__h*(Tau + tau_syn_in)/(Tau*tau_syn_in))/(C_m*(Tau - tau_syn_in))", 
    "__P_I_shape_in__2_2": "exp(-__h/Tau)", 
    "__P_I_shape_ex__1_1": "exp(-__h/tau_syn_ex)", 
    "__P_I_shape_ex__0_0": "exp(-__h/tau_syn_ex)", 
    "__P_I_shape_ex__1_0": "__h*exp(-__h/tau_syn_ex)"
  }
}
```

### Conductance-based model
The following example shows an input model that corresponds to a conductance-based integrate-and-fire neuron with an alpha-shaped postsynaptic response. This example provides also a `parameters` dictionary for the stiffness testing.

```
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
      "initial_values": ["0",  "e / tau_syn_ex"]
    }
  ],

  "odes": [
    {
      "symbol": "V_m",
      "definition": "(-(g_L*(V_m-E_L))-(g_ex*(V_m-E_ex))-(g_in*(V_m-E_in))+I_stim+I_e)/C_m",
      "initial_values": ["E_L"],
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
```
{
  "shape_state_variables": [
    "g_in__d", 
    "g_in", 
    "g_ex__d", 
    "g_ex"
  ], 
  "shape_initial_values": [
    "0", 
    "e/tau_syn_in", 
    "0", 
    "e/tau_syn_ex"
  ], 
  "shape_ode_definitions": [
    "-1/tau_syn_in**2 * g_in + -2/tau_syn_in * g_in__d", 
    "-1/tau_syn_ex**2 * g_ex + -2/tau_syn_ex * g_ex__d"
  ], 
  "solver": "numeric",
  "ode_system_type": "stiff"
}
```
