from ignition.sfl.language import *
from ignition.sfl.generators import generate

u = Variable('u', dim=3, space='L2')
K = Coefficient('K', rank=0)

expr = div(K * grad(u))


"""Definition of poisson manufactured solution"""

import numpy as np

class velEx(object):
    """Computes exact 'Darcy' velocity"""
    def __init__(self,duex,aex):
        self.duex = duex
        self.aex = aex

    def uOfX(self,X):
        du = self.duex.duOfX(X)
        A  = np.reshape(self.aex(X),(3,3))
        return -np.dot(A,du)

    def uOfXT(self,X,T):
        return self.uOfX(X)


class u5Ex(object):
    """'Manufactured' analytical solution"""
    def uOfX(self,x):
        return x[0]**2+x[1]**2+x[2]**2

    def uOfXT(self,X,T):
        return self.uOfX(X)

    def duOfX(self,X):
        du = 2.0*np.reshape(X[0:3],(3,))
        return du

    def duOfXT(self,X,T):
        return self.duOfX(X)


##################################################
#define coefficients a(x)=[a_{ij}] i,j=0,2, right hand side f(x)  and analytical solution u(x)
#u = x*x + y*y + z*z, a_00 = x + 5, a_11 = y + 5.0 + a_22 = z + 10.0
#f = -2*x -2*(5+x) -2*y-2*(5+y) -2*z-2*(10+z)
#
def a5(x):
    return np.array([[x[0] + 5.0,0.0,0.0],[0.0,x[1] + 5.0,0.0],[0.0,0.0,x[2]+10.0]],'d')

def f5(x):
    return -2.0*x[0] -2*(5.+x[0]) -2.*x[1]-2.*(5.+x[1]) -2.*x[2]-2.*(10+x[2])

def getDBC5(x, flag):
    """Dirichlet boundary condition functions

    Defined as (x=0,y,z), (x,y=0,z), (x,y=1,z), (x,y,z=0), (x,y,z=1)
    """
    if x[0] in [0.0] or x[1] in [0.0,1.0] or x[2] in [0.0,1.0]:
        return lambda x,t: u5Ex().uOfXT(x,t)

def getAdvFluxBC5(x, flag):
    pass

def getDiffFluxBC5(x, flag):
    """Specifies flux on (x=1,y,z)"""
    if x[0] == 1.0:
        n = np.zeros((nd,),'d'); n[0]=1.0
        return lambda x,t: np.dot(velEx(u5Ex(), a5).uOfXT(x,t),n)
    if not (x[0] in [0.0] or x[1] in [0.0,1.0] or x[2] in [0.0,1.0]):
        return lambda x,t: 0.0

# Necessary dictionaries for problem class
#store a,f in dictionaries since coefficients class allows for one entry per component
proteus_problem_kws = {
    "aOfX" : {0 : a5},
    "fOfX" : {0 : f5},
    "analyticalSolution" : {0 : u5Ex()},
    "analyticalSolutionVelocity" : {0 : velEx(analyticalSolution[0], aOfX[0])},    
    "dirichletConditions" : {0 : getDBC5},
    "advectiveFluxBoundaryConditions" : {0 : getAdvFluxBC5},
    "diffusiveFluxBoundaryConditions" : {0 : {0 : getDiffFluxBC5}},
    "fluxBoundaryConditions" : {0:'setFlow'}, # options are 'setFlow','noFlow','mixedFlow'
}

proteus_numerics_kws = {
    "femSpaces" : {0:C0_AffineLinearOnSimplexWithNodalBasis}
    "elementQuadrature" : SimplexGaussQuadrature(nd,4)
    "elementBoundaryQuadrature" : SimplexGaussQuadrature(nd-1,4)
    "nnx" : 7
    "nny" : 7
    "nnz" : 7
    "levelNonlinearSolver" : Newton
    "maxNonlinearIts" : 2
    "maxLineSearches" : 1
    "nl_atol_res" : 1.0e-8
    "tolFac" : 0.0
    "matrix" : SparseMatrix
    "parallel" : False
    "multilevelLinearSolver" : LU
    "levelLinearSolver" : LU
    "numericalFluxType" : Advection_DiagonalUpwind_Diffusion_IIPG_exterior
    "linTolFac" : 0.0
    "l_atol_res" : 1.0e-10
}

#generate('proteus', expr)
from ignition.proteus.runner import proteus_runner
proteus_runner(expr, proteus_problem_kws, proteus_numerics_kws, sys.argv[1:], 
               coefficient_routine=TransportCoefficients.PoissonEquationCoefficients,
               coefficient_args=("aOfX", "fOfX", "nc", "nd"))
