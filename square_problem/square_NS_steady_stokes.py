from firedrake import *
from firedrake.output import VTKFile
import numpy as np

# ---------------------------------------
# here we try to reconstruct the problem solved by Braack
# that is, STEADY flow
# f(x,y) = (sin(x) + sin(y), 0)
# --------------------------------------


# mesh
mesh = Mesh("square.msh")                    # upload file, .msh created using Gmsh

# finite element space (Taylor-Hood: P2 + P1)
V = VectorFunctionSpace(mesh, "CG", 2)              # vector space for u
P = FunctionSpace(mesh, "CG", 1)                    # scalar space for p
W = V * P                                           # mixed space for (u,p)
x, y = SpatialCoordinate(mesh)                                # set x, y as coordinates

# the problem
# h = as_vector((0, 0))
f = as_vector((sin(x) + sin(y), 0))
# v_r = as_vector([0, 0])
g = as_vector((0, 0))                           # set (0,0) value (later used at u = (0,0) on boundaries)
k = 1 # 1, 2, 3, 4
nu = Constant(5 * 10**(-k))

# boundary conditions
bc_walls = DirichletBC(W.sub(0), g, 5)        # W.sub(0) gives me velocity (.sub(1) is pressure), (0,0), no. 10 based on the .msh (.geo) file is the walls

# auxilliary definitions
bcs = [bc_walls]                # put together all boundaries
n = FacetNormal(mesh)                               # normal vector - will be determined in the integral
I = Identity(mesh.geometric_dimension())            # identity matrix - will be used in the stress tensor

# ------------------------------------------------------------------------------------------------------------------
# VARIATIONAL PROBLEM
# ------------------------------------------------------------------------------------------------------------------

# functions for CURRENT time step
w0 = Function(W)                                    # w0 is the mixed function of v0 + p0  
v0, p0 = split(w0)                                  # v0, p0 separately

# funcitons for NEXT time step
w = Function(W)                                     # w is the mixed function of v + p  
v, p = split(w)                                     # v, p separately

# test functions
v_, p_ = TestFunctions(W)                           # (v_, p_) tuple

# viscous term
def a(u, phi):                                                                 
    a = (inner(nu*grad(u), grad(phi))) * dx
    return a         

# pressure and incompressibility
def b(q, u):                                         # p * div(v_) // p_ * div(v)
    return q * div(u) * dx                           # bilinear form

F = a(v, v_) - b(p_, v) - b(p, v_) - - inner(f, v_) * dx


# ------------------------------------------------------------------------------------------------------------------
# SOLVER
# ------------------------------------------------------------------------------------------------------------------

# parameters for solver
prob = NonlinearVariationalProblem(F, w, bcs=bcs)                # (weak formulation, weak solution, boundary conditions)
solver = NonlinearVariationalSolver(prob, solver_parameters={
    'snes_type': 'newtonls',                                     # Newton's method F(w)=0
    'snes_linesearch_type': 'bt',                                # add line-search
    'snes_atol': 1e-12,                                          # error tolerance
    'snes_rtol': 1e-12,                                          # residuum tolerance
    'snes_max_it': 20,                                           # max. iterations for one time step
    'ksp_type': 'preonly',                                       # use preconditioning           
    'pc_type': 'lu',                                             # use LU for precon
    'pc_factor_mat_solver_type': 'mumps'                         # direct solver
})

# output files
out_v = VTKFile(f"results/v_sqr_steady_stokes_{k}.pvd")              # result of velocity
out_p = VTKFile(f"results/p_sqr_steady_stokes_{k}.pvd")              # result of pressure

solver.solve()
v_sol, p_sol = w.subfunctions 
out_v.write(v_sol)
out_p.write(p_sol)

if COMM_WORLD.rank == 0:                         # for the main process (for MPI):
    print(f"Finished computations.") 
    print(f"- saved to results/v_sqr_steady_stokes_{k}.pvd")
    print(f"- saved to results/p_sqr_steady_stokes_{k}.pvd")