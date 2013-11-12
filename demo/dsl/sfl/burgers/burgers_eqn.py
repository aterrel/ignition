from ignition.dsl.sfl.language import *
from ignition.utils.proteus.coefficient import sfl_coefficient

from proteus.TransportCoefficients import ViscousBurgersEqn

u = Variable('u')
v, nu = Constants('v nu')

f = u**2
eqn = Dt(u) + div(v*f - nu*grad(u))
strong_form = StrongForm(eqn)

burger_coefficients = sfl_coefficient(strong_form,
                                      base_class=ViscousBurgersEqn,
                                      gen_evaluate=False)
