from sympy import *
from sympy.matrices import zeros

__a__, __h__, delta = symbols('__a__ __h__ delta')
<#compress>
    <#list variables as variable> ${variable.getName()} , </#list> = symbols('<#list variables as variable> ${variable.getName()} </#list>')
</#compress>

# Handle aliases
<#list aliases as alias>
${alias.getName()} = ${printer.print(alias.getDeclaringExpression().get())}
</#list>

# Shapes must be symbolic for the differetiation step
rhsTmp = ${printer.print(ode.getRhs())}
constantInputs = simplify(1/diff(rhsTmp, ${EQs[0].getLhs()}) * (rhsTmp - diff(rhsTmp, ${ode.getLhs()})*${ode.getLhs()}) - (
<#assign operator = "">
<#compress> <#list EQs as eq>
${operator} ${eq.getLhs()}
<#assign operator = "+">
</#list> </#compress>
))

# TODO take the comment for the source model
<#list EQs as eq>
${eq.getLhs()} = ${printer.print(eq.getRhs())}
</#list>
rhs = ${printer.print(ode.getRhs())}
dev${ode.getLhs()} = diff(rhs, ${ode.getLhs()})
dev_t_dev${ode.getLhs()} = diff(dev${ode.getLhs()}, t)

if dev_t_dev${ode.getLhs()} == 0:
    solverType = open('solverType.tmp', 'w')
    solverType.write("exact")

    propagatorStepFile = open('propagator.step.tmp', 'w')
    propagatorStepFile.write("${ode.getLhs()} = P30 * (" + str(constantInputs) + ") ")
    # calculate -1/Tau
    c1 = diff(rhs, V)
    # The symbol must be declared again. Otherwise, the right hand side will be used for the derivative
    ${EQs[0].getLhs()} = symbols("${EQs[0].getLhs()}")
    c2 = diff( ${printer.print(ode.getRhs())} , ${EQs[0].getLhs()})
    f = open('P30.tmp', 'w')
    f.write("P30 real = " + str(simplify(c2 / c1 * (exp(__h__ * c1) - 1))) + "# P00 expression")