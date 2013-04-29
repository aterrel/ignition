class ProteusProblem(object):
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
    LevelModelType = OneLevelTransport
