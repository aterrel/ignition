from ignition.dsl.sfl.language import *
from ignition.utils.proteus.coefficient import sfl_coefficient

from proteus.TransportCoefficients import ViscousBurgersEqn

# Constants
uL = 2.0
uR = 1.0
v_val = [1.0]
nu_val = 1e-6
nd = 1
T = 0.0015

class RiemIC:
    def __init__(self, dbc):
        self.uLeft = dbc([0.0, 0.0, 0.0])([0.0, 0.0, 0.0], 0.0)
        self.uRight = uR

    def uOfXT(self, x, t):
        if x[0] <= 0.5: #0.0
            return self.uLeft
        else:
            return self.uRight


def getDBC(x):
    if x[0] == 0.0:
        return lambda x,t: uL


# Define Equations
u = Variable('u')
v, nu = Constants('v nu')

f = u**2
eqn = Dt(u) + div(v*f - nu*grad(u))
strong_form = StrongForm(eqn)

burger_coefficients = sfl_coefficient(strong_form,
                                      base_class=ViscousBurgersEqn,
                                      gen_evaluate=False,
                                      v=v_val, nu=nu_val, nd=nd)

#analyticalSolution = {0:RiemIC(getDBC)}
initialConditions = {0: RiemIC(getDBC)}
dirichletConditions = {0: getDBC}
fluxBoundaryConditions = {0: 'outFlow'}
advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}

