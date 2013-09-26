"""Script printer for Proteus"""

from ignition.utils import indent_code

from sfl_printer import SFLPrinter


script_header =  """
\"\"\"test_poisson.py [options]

Solves the Heterogeneous Poisson equation on a unit cube. A full
script for testing generation and tools provided by proteus.
\"\"\"

import numpy as np
import sys

from proteus import Comm, Profiling, NumericalSolution, TransportCoefficients, default_so, default_s
from proteus.FemTools import C0_AffineLinearOnSimplexWithNodalBasis
from proteus.LinearSolvers import LU
from proteus.NonlinearSolvers import Newton
from proteus.NumericalFlux import Advection_DiagonalUpwind_Diffusion_IIPG_exterior
from proteus.Quadrature import SimplexGaussQuadrature
from proteus.superluWrappers import SparseMatrix
from proteus.TimeIntegration import NoIntegration

from ignition.utils.proteus.defaults import ProteusProblem, ProteusNumerics
from ignition.utils.proteus.optparser import get_prog_opts


log = Profiling.logEvent
nd = %{num_dimension}d
"""

problem_template = """
class %{name}s(ProteusProblem):

    def __init__(self):
        self.name = %{name}s
        self.nd = %{num_dimension}d # space dimension
        self.nc = %{num_components}d  # one component

        %{problem_settings}s

        #load analytical solution, dirichlet conditions, flux boundary conditions into the expected variables
        self.analyticalSolution = analyticalSolution
        self.analyticalSolutionVelocity = analyticalSolutionVelocity

        self.dirichletConditions = dirichletConditions
        self.advectiveFluxBoundaryConditions = advectiveFluxBoundaryConditions
        self.diffusiveFluxBoundaryConditions = diffusiveFluxBoundaryConditions
        self.fluxBoundaryConditions = fluxBoundaryConditions

        %{problem_coefficents}s
"""

numeric_template = """
class C0P1_Poisson_Numerics(ProteusNumerics):
    #steady-state so no time integration
    timeIntegration = NoIntegration
    #number of output timesteps
    nDTout = 1

    #finite element spaces
    femSpaces = {0:C0_AffineLinearOnSimplexWithNodalBasis}
    #numerical quadrature choices
    elementQuadrature = SimplexGaussQuadrature(nd,4)
    elementBoundaryQuadrature = SimplexGaussQuadrature(nd-1,4)

    #number of nodes in x,y,z
    nnx = 7
    nny = 7
    nnz = 7
    #if unstructured would need triangleOptions flag to be set


    #number of levels in mesh
    nLevels = 1

    #no stabilization or shock capturing
    subgridError = None

    shockCapturing = None

    #nonlinear solver choices
    multilevelNonlinearSolver  = Newton
    levelNonlinearSolver = Newton
    #linear problem so force 1 iteration allowed
    maxNonlinearIts = 2
    maxLineSearches = 1
    fullNewtonFlag = True
    #absolute nonlinear solver residual tolerance
    nl_atol_res = 1.0e-8
    #relative nonlinear solver convergence tolerance as a function of h
    #(i.e., tighten relative convergence test as we refine)
    tolFac = 0.0

    #matrix type
    matrix = SparseMatrix

    #convenience flag
    parallel = False

    if parallel:
        multilevelLinearSolver = KSP_petsc4py
        #for petsc do things lie
        #"-ksp_type cg -pc_type asm -pc_asm_type basic -ksp_atol  1.0e-10 -ksp_rtol 1.0e-10 -ksp_monitor_draw" or
        #-pc_type lu -pc_factor_mat_solver_package
        #can also set -pc_asm_overlap 2 with default asm type (restrict)
        levelLinearSolver = KSP_petsc4py#
        #for petsc do things like
        #"-ksp_type cg -pc_type asm -pc_asm_type basic -ksp_atol  1.0e-10 -ksp_rtol 1.0e-10 -ksp_monitor_draw" or
        #-pc_type lu -pc_factor_mat_solver_package
        #can also set -pc_asm_overlap 2 with default asm type (restrict)
        #levelLinearSolver = PETSc#
        #pick number of layers to use in overlap
        nLayersOfOverlapForParallel = 0
        #type of partition
        parallelPartitioningType = MeshParallelPartitioningTypes.node
        #parallelPartitioningType = MeshParallelPartitioningTypes.element
        #have to have a numerical flux in parallel
        numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior
        #for true residual test
        linearSolverConvergenceTest = 'r-true'
        #to allow multiple models to set different ksp options
        #linear_solver_options_prefix = 'poisson_'
        linearSmoother = None
    else:
        multilevelLinearSolver = LU
        levelLinearSolver = LU
        numericalFluxType = Advection_DiagonalUpwind_Diffusion_IIPG_exterior

    #linear solver relative convergence test
    linTolFac = 0.0
    #linear solver absolute convergence test
    l_atol_res = 1.0e-10

    #conservativeFlux =  {0:'pwl'}

"""

script_foot_template = """
def init_mpi_petsc(opts):
    log("Initializing MPI")
    if opts.petscOptions != None:
        petsc_argv = sys.argv[:1]+opts.petscOptions.split()
        log("PETSc options from commandline")
        log(str(petsc_argv))
    else:
        petsc_argv=sys.argv[:1]
    if opts.petscOptionsFile != None:
        petsc_argv=[sys.argv[0]]
        petsc_argv += open(opts.petscOptionsFile).read().split()
        log("PETSc options from commandline")
        log(str(petsc_argv))
    return Comm.init(argv=petsc_argv)

def main(*args):
    opts, args = get_prog_opts(args, __doc__)
    comm = init_mpi_petsc(opts)
    problem_list = [Poisson(),]
    simulation_list = [default_s]
    numerics_list = [C0P1_Poisson_Numerics(),]
    numerics_list[0].periodicDirichletConditions = problem_list[0].periodicDirichletConditions
    numerics_list[0].T = problem_list[0].T
    simulation_name = problem_list[0].name + "_" + numerics_list[0].__class__.__name__
    simulation_name_proc = simulation_name + "_" + repr(comm.rank())
    simFlagsList = [{ 'simulationName': simulation_name,
                      'simulationNameProc': simulation_name_proc,
                      'dataFile': simulation_name_proc + '.dat',
                      'components' : [ci for ci in range(problem_list[0].coefficients.nc)],
                      }]

    so = default_so
    so.name = problem_list[0].name
    so.pnList = problem_list
    so.sList = [default_s]
    try:
        so.systemStepControllerType = numerics_list[0].systemStepControllerType
    except AttributeError:
        pass
    try:
        so.tnList = numerics_list[0].tnList
        so.archiveFlag = numerics_list[0].archiveFlag
    except AttributeError:
        pass

    runNumber = 0
    runName = so.name + repr(runNumber)
    Profiling.procID=comm.rank()
    if simulation_list[0].logAllProcesses or opts.logAllProcesses:
        Profiling.logAllProcesses = True
    Profiling.flushBuffer=simulation_list[0].flushBuffer

    if opts.logLevel > 0:
        Profiling.openLog(runName+".log",opts.logLevel)


    ns = NumericalSolution.NS_base(default_so, problem_list, numerics_list, simulation_list,
                                   opts, simFlagsList)

    ns.calculateSolution(runName)

if __name__ == "__main__":
    main(sys.argv[1:])

"""


class ProteusScriptPrinter(SFLPrinter):
    comment_str = "#"

    def __init__(self, generator):
        self._generator = generator

    def _print_header(self):
        return script_header

    def _print_problem_class(self):
        ret_code = problem_template
        return ret_code

    def _print_manufactured_solution(self):
        ret_code = self.generator.kwargs.get('manufactured_solution_module', '')
        if ret_code:
            ret_code = "from %s import *\n\n" % ret_code
        return ret_code

    def _print_numeric_class(self):
        ret_code = numeric_template
        return ret_code

    def _print_script_footer(self):
        ret_code = script_foot_template
        return ret_code

    def print_file(self):
        ret_code = ""
        ret_code += self._print_header()
        ret_code += self._print_manufactured_solution()
        ret_code += self._print_problem_class()
        ret_code += self._print_numeric_class()
        ret_code += self._print_script_footer()
        return ret_code

    
