`# ODE Toolbox - a framework for automatic solver selection for systems of differential equations 

## Prerequisites 
* ode-toolbox requires `sympy` in the version of at least
* For the stifness testing an installation of `PyGSL` is necessary (http://pygsl.sourceforge.net/). If `PyGSL` is not installed than the stiffness testing is omitted but the remainder of the analysis framework is still working.

## Usage
The `ode-toolbox` can be used either as a normal python module (cf.  tests in `tests` to see the API) or as command line application. The input for `ode-toolbox` is stored in `json`-files (file format will be explained in the next section). The entry-point for the analysis is `ode_analyzer.py`, The script expects the name of a JSON file as its only command line argument:

```
python ode_analyzer.py iaf_cond_alpha.json
```
`ode_analyzer.py` stores output in an JSON-file named `result-$date$.json` where `$date$` correspond to the current date.
## Input


## Output

## Example