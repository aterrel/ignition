import optparse
import sys
from ignition.dsl.sfl.generators import ProteusCoefficientGenerator

from proteus import Comm, Profiling, NumericalSolution, default_so, default_s

log = Profiling.logEvent


def proteus_runner(expr, problem_kws, numerics_kws, *args, **kws):
    opts, args = get_prog_opts(args, __doc__)
    log = kws.get('log', None)
    if log is None: log = Profiling.logEvent
    comm = init_mpi_petsc(opts, log)
    problem_list = [ProteusProblem(**problem_kws),]
    simulation_list = [default_s]
    numerics_list = [ProteusNumerics(**numerics_kws),]
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


def init_mpi_petsc(opts, log):
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


class ProteusBase(object):

    def __init__(self, **kws):
        for k, v in kws.iteritems():
            setattr(self, k, v)


class ProteusProblem(ProteusBase):
    """A default Problem for Proteus """

    name = None # Name of model, None or string
    nd = 1 # Number of spatial dimensions of the model domain
    domain = None # None or proteus.Domain.D_base
    movingDomain=False
    polyfile = None
    meshfile = None
    genMesh = True
    L=(1.0,1.0,1.0) # Tuple of dimensions for simple box shaped domain
    analyticalSolution = {} #Dictionary of analytical solutions for each component
    coefficients = None
    dirichletConditions = {}
    periodicDirichletConditions = None
    fluxBoundaryConditions = {} # Dictionary of flux boundary condition flags for each component
                                # ('outflow','noflow','setflow','mixedflow')
    advectiveFluxBoundaryConditions =  {} # Dictionary of advective flux boundary conditions setter functions
    diffusiveFluxBoundaryConditions = {} # Dictionary of diffusive flux boundary conditions setter functions
    stressFluxBoundaryConditions = {} # Dictionary of stress tensor flux boundary conditions setter functions
    initialConditions = None #Dictionary of initial condition function objects
    weakDirichletConditions = None # Dictionary of weak Dirichlet constraint setters
    bcsTimeDependent = True # Allow optimizations if boundary conditions are not time dependent
    dummyInitialConditions = False #mwf temporary hack for RD level sets
    finalizeStep = lambda c: None
    T=1.0 # End of time interval
    sd = True # Use sparse representation of diffusion tensors
    LevelModelType = proteus.Transport.OneLevelTransport


class ProteusNumerics(ProteusBase):
    """The default values for numerics modules
    """
    stepController = proteus.StepControl.FixedStep # The step controller class derived from :class:`proteus.StepControl.SC_base`
    timeIntegration = proteus.TimeIntegration.NoIntegration # The time integration class derived from :class:`proteus.TimeIntegraction.TI_base
    timeIntegrator  = proteus.TimeIntegration.ForwardIntegrator # Deprecated, the time integrator class
    runCFL = 0.9 # The maximum CFL for the time step
    nStagesTime = 1 # The number of stages for the time discretization
    timeOrder= 1 # The order of the time discretization
    DT = 1.0 # The time step
    nDTout = 1 # The number of output time steps
    rtol_u = {0:1.0e-4} # A dictionary of relative time integration tolerances for the components
    atol_u = {0:1.0e-4} # A dictionary of absolute time integration tolerances for the components
    nltol_u = 0.33 # The nonlinear tolerance factor for the component error
    ltol_u = 0.05 # The linear tolerance factor for the component error
    rtol_res = {0:1.0e-4} # A dictionary of relative tolerances for the weak residuals
    atol_res = {0:1.0e-4} # A dictionary of absolute tolerances for the weak residuals
    nl_atol_res = 1.0 # The nonlinear residual tolerance
    l_atol_res = 1.0 # The linear residual tolerance
    femSpaces = {} # A dictionary of the finite element classes for each component
                   # The classes should be of type :class:`proteus.FemTools.ParametricFiniteElementSpace`
    elementQuadrature = None # A quadrature object for element integrals
    elementBoundaryQuadrature = None # A quadrature object for element boundary integrals
    nn = 3 # Number of nodes in each direction for regular grids
    nnx = None # Number of nodes in the x-direction for regular grids
    nny = None # Number of nodes in the y-direction for regular grids
    nnz = None # Number of nodes in the z-direction for regular grids
    triangleOptions="q30DenA" # Options string for triangle or tetGen
    nLevels = 1 # Number of levels for multilevel mesh
    subgridError = None # The subgrid error object of a type derived from :class:`proteus.SubgridError.SGE_base`
    massLumping = False # Boolean to lump mass matrix
    reactionLumping = False # Boolean to lump reaction term
    shockCapturing = None # The shock capturing diffusion object of a type derived from :class:`proteus.ShockCapturing.SC_base`
    numericalFluxType = None # A numerical flux class of type :class:`proteus.NumericalFlux.NF_base`
    multilevelNonlinearSolver  = proteus.NonlinearSolvers.NLNI # A multilevel nonlinear solver class of type :class:`proteus.NonlinearSolvers.MultilevelNonlinearSolver`
    levelNonlinearSolver = proteus.NonlinearSolvers.Newton # A nonlinear solver class of type :class:`proteus.NonlinearSolvers.NonlinearSolver`
    nonlinearSmoother = proteus.NonlinearSolvers.NLGaussSeidel # A nonlinear solver class of type :class:`proteus.NonlinearSolvers.NonlinearSolver`
    fullNewtonFlag = True # Boolean to do full Newton or modified Newton
    nonlinearSolverNorm = staticmethod(proteus.LinearAlgebraTools.l2Norm) # Norm to use for nonlinear algebraic residual
    tolFac = 0.01
    atol = 1.0e-8
    maxNonlinearIts =10
    maxLineSearches =10
    psitc = {'nStepsForce':3,'nStepsMax':100}
    matrix = proteus.superluWrappers.SparseMatrix
    multilevelLinearSolver = proteus.LinearSolvers.LU
    levelLinearSolver = proteus.LinearSolvers.LU
    computeEigenvalues = False
    computeEigenvectors = None #'left','right'
    linearSmoother = proteus.LinearSolvers.StarILU #GaussSeidel
    linTolFac = 0.001
    conservativeFlux = None
    checkMass = False
    multigridCycles = 2
    preSmooths = 2
    postSmooths = 2
    computeLinearSolverRates = False
    printLinearSolverInfo = False
    computeLevelLinearSolverRates = False
    printLevelLinearSolverInfo = False
    computeLinearSmootherRates = False
    printLinearSmootherInfo = False
    linearSolverMaxIts = 1000
    linearWCycles = 3
    linearPreSmooths = 3
    linearPostSmooths = 3
    computeNonlinearSolverRates=True
    printNonlinearSolverInfo=False
    computeNonlinearLevelSolverRates=False
    printNonlinearLevelSolverInfo=False
    computeNonlinearSmootherRates=False
    printNonlinearSmootherInfo=False
    nonlinearPreSmooths=3
    nonlinearPostSmooths=3
    nonlinearWCycles=3
    useEisenstatWalker=False
    maxErrorFailures=10
    maxSolverFailures=10
    needEBQ_GLOBAL = False
    needEBQ = False
    auxiliaryVariables=[]
    restrictFineSolutionToAllMeshes=False
    parallelPartitioningType = proteus.MeshTools.MeshParallelPartitioningTypes.element
    #default number of layers to use > 1 with element partition means
    #C0P1 methods don't need to do communication in global element assembly
    #nodal partitioning does not need communication for C0P1 (has overlap 1) regardless
    nLayersOfOverlapForParallel = 1
    parallelPeriodic=False#set this to true and use element,0 overlap to use periodic BC's in parallel
    nonlinearSolverConvergenceTest = 'r'
    levelNonlinearSolverConvergenceTest = 'r'
    linearSolverConvergenceTest = 'r' #r,its,r-true for true residual
    #we can add this if desired for setting solver specific options in petsc
    #linear_solver_options_prefix= None #


def get_prog_opts(args, usage=""):
    """Returns options and unused args from command line arg list.

    usage - optional argurment for help option.
    """

    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-I", "--inspect",
                      help="Inspect namespace at 't0','user_step'",
                      action="store",
                      dest="inspect",
                      default='')
    parser.add_option("-i", "--interactive",
                      help="Read input from stdin",
                      action="store_true",
                      dest="interactive",
                      default='')
    parser.add_option("-d", "--debug",
                      help="start the python debugger",
                      action="store_true",
                      dest="debug",
                      default=False)
    parser.add_option("-V", "--viewer",
                      help="Set the method to use for runtime viewing. Can be vtk or gnuplot",
                      action="store",
                      type="string",
                      dest="viewer",
                      default=False)
    parser.add_option("-C", "--plot-coefficients",
                      help="Plot the coefficients of the transport models",
                      action="store_true",
                      dest="plotCoefficients",
                      default=False)
    parser.add_option("-P", "--petsc-options",
                      help="Options to pass to PETSc",
                      action="store",
                      type="string",
                      dest="petscOptions",
                      default=None)
    parser.add_option("-O", "--petsc-options-file",
                      help="Text file of ptions to pass to PETSc",
                      action="store",
                      type="string",
                      dest="petscOptionsFile",
                      default=None)
    parser.add_option("-D", "--dataDir",
                      help="Options to pass to PETSc",
                      action="store",
                      type="string",
                      dest="dataDir",
                      default='')
    parser.add_option("-b", "--batchFile",
                      help="Read input from a file",
                      action="store",
                      type="string",
                      dest="batchFileName",
                      default="")
    parser.add_option("-p", "--profile",
                      help="Generate a profile of the  run",
                      action="store_true",
                      dest="profile",
                      default=False)
    parser.add_option("-T", "--useTextArchive",
                      help="Archive data in ASCII text files",
                      action="store_true",
                      dest="useTextArchive",
                      default=False)
    parser.add_option("-m", "--memory",
                      help="Track memory usage of the  run",
                      action="callback",
                      callback=Profiling.memProfOn_callback)
    parser.add_option("-M", "--memoryHardLimit",
                      help="Abort program if you reach the per-MPI-process memory hardlimit (in GB)",
                      action="callback",
                      type="float",
                      callback=Profiling.memHardLimitOn_callback,
                      default = -1.0,
                      dest = "memHardLimit")
    parser.add_option("-l", "--log",
                      help="Store information about what the code is doing,0=none,10=everything",
                      action="store",
                      type="int",
                      dest="logLevel",
                      default=1)
    parser.add_option("-A", "--logAllProcesses",
                      help="Log events from every MPI process",
                      action="store_true",
                      dest="logAllProcesses",
                      default=False)
    parser.add_option("-v", "--verbose",
                      help="Print logging information to standard out",
                      action="callback",
                      callback=Profiling.verboseOn_callback)
    parser.add_option("-E", "--ensight",
                      help="write data in ensight format",
                      action="store_true",
                      dest="ensight",
                      default=False)
    parser.add_option("-L", "--viewLevels",
                      help="view solution on every level",
                      action="store_true",
                      dest="viewLevels",
                      default=False)
    parser.add_option("--viewMesh",
                      help="view mesh",
                      action="store_true",
                      dest="viewMesh",
                      default=False)
    parser.add_option("-w", "--wait",
                      help="stop after each nonlinear solver call",
                      action="store_true",
                      dest="wait",
                      default=False)
    parser.add_option('--probDir',
                      default='.',
                      help="""where to find problem descriptions""")
    parser.add_option("-c","--cacheArchive",
                      default=False,
                      dest="cacheArchive",
                      action="store_true",
                      help="""don't flush the data files after each save, (fast but may leave data unreadable)""")
    parser.add_option("-G","--gatherArchive",
                      default=False,
                      dest="gatherArchive",
                      action="store_true",
                      help="""collect data files into single file at end of simulation (convenient but slow on big run)""")

    parser.add_option("-H","--hotStart",
                      default=False,
                      dest="hotStart",
                      action="store_true",
                      help="""Use the last step in the archive as the intial condition and continue appending to the archive""")
    parser.add_option("-B","--writeVelocityPostProcessor",
                      default=False,
                      dest="writeVPP",
                      action="store_true",
                      help="""Use the last step in the archive as the intial condition and continue appending to the archive""")

    opts, args = parser.parse_args()
    return opts, args


def sfl_coefficient(strong_form, *args, **kws):
    """Generates a proteus transport coefficient class from sfl and returns an
    instance of the class

    args and kws are passed to the constructor of that instance.
    """
    generator = ProteusCoefficientGenerator(strong_form)
    generator.to_file()
    sys.path.append(generator.module_path)
    mod = __import__(generator.module_name)
    coeff_instance = getattr(mod, generator.classname)(*args, **kws)
    coeff_instance.strong_form = strong_form
    return coeff_instance
