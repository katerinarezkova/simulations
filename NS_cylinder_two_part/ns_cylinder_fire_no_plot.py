import numpy as np
from pyop2.mpi import COMM_WORLD


from firedrake import *
from firedrake.output import VTKFile
import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# TASK
# ------------------------------------------------------------------------------------------------------------------

# Recreation of the N-S flow around a cylinder as in
# https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html.
# Used:
#       - walls: no-slip dirichlet
#       - inflow: parabolic U = 1.5
#       - time: Crank-Nicolson scheme
# Reylonds:
#       - U_mean = 1
#       - L = 0.1
#       - nu = 0.001
#       - Re = U_mean * L / nu
#            = 100
# Solve for:
#       - u, p 
#       - drag, lift force

# ------------------------------------------------------------------------------------------------------------------
# DOMAIN
# ------------------------------------------------------------------------------------------------------------------

# mesh
mesh = Mesh("mesh_cylinder.msh")                    # upload file, .msh created using Gmsh

# spaces (Taylor-Hood: P2 + P1)
V = VectorFunctionSpace(mesh, "CG", 2)              # vector space for u
P = FunctionSpace(mesh, "CG", 1)                    # scalar space for p
W = V * P                                           # mixed space for (u,p)

# parameters: Constant for physical values that can be changed. Then, nu.assign(0.002)
U = 1.5                                             # maximal velocity at inflow
Y = 0.41                                            # height of the tunnel for inflow velocity - check from .msh
r = 0.05                                            # diameter of the object perpendicular to the flow (cylinder) - check from .msh
nu = Constant(0.001)                                # viscosity
dt = 0.1                                            # time step
t_end = 5                                           # maximal time
theta = Constant(0.5)                               # for time-schemes (Crank-Nicolson)

# boundary conditions
noslip = Constant((0, 0))                           # set (0,0) value (later used at u = (0,0) on boundaries)
bc_walls = DirichletBC(W.sub(0), noslip, 10)        # W.sub(0) gives me velocity (.sub(1) is pressure), (0,0), no. 10 based on the .msh (.geo) file is the walls
bc_cylinder = DirichletBC(W.sub(0), noslip, 9)      # W.sub(0) gives me velocity (.sub(1) is pressure), (0,0), no. 9 based on the .msh (.geo) file is the cylinder

# inflow
x, y = SpatialCoordinate(mesh)                                # set x, y as coordinates
v_in = as_vector([U * 4.0 * y * (Y - y) / (Y**2), 0.0])       # parabolic inflow
bc_in = DirichletBC(W.sub(0), v_in, 7)                        # assign velocity at inflow to the correct boundary (7)

# auxilliary definitions
bcs = [bc_cylinder, bc_walls, bc_in]                # put together all boundaries
n = FacetNormal(mesh)                               # normal vector - will be determined in the integral
I = Identity(mesh.geometric_dimension())            # identity matrix - will be used in the stress tensor
U_mean = 2/3 * U                                    # mean value of velocity for parabolic flow (= 1)
L = 2 * r                                           # characteristic length of flow (= 0.1)
Re = U_mean * L / nu                                # Reynolds number (= 100)

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

# convective and viscous term
def a(u, v_conv, phi):                                                             # u = solve, v_conv = fix, phi = test function
    D = sym(grad(u))                                                               # strain-rate tensor D (for physical properties)
    return (dot(dot(grad(u), v_conv), phi) + inner(2*nu*D, grad(phi))) * dx        # trilinear form

# pressure and incompressibility
def b(q, u):                                         # p * div(v_) // p_ * div(v)
    return q * div(u) * dx                           # bilinear form

# time discretization
F1 = a(v, v, v_) - b(p_, v) - b(p, v_)               # RHS for NEXT time step
F0 = a(v0, v0, v_) - b(p_, v0) - b(p, v_)            # RHS for CURRENT time step (notice p is unchanged - time derivation in u)
Ft = dot(v - v0, v_) * dx                            # transient part

# whole variational form
F = (1.0/dt) * Ft + theta * F1 + (1.0 - theta) * F0  # theta = 0.5 --> Crank_Nicolson

# ------------------------------------------------------------------------------------------------------------------
# SOLVER
# ------------------------------------------------------------------------------------------------------------------

# parameters for solver
prob = NonlinearVariationalProblem(F, w, bcs=bcs)                # (weak formulation, weak solution, boundary conditions)
solver = NonlinearVariationalSolver(prob, solver_parameters={
    'snes_type': 'newtonls',                                     # Newton's method F(w)=0
    'snes_atol': 1e-12,                                          # error tolerance
    'snes_rtol': 1e-12,                                          # residuum tolerance
    'snes_max_it': 20,                                           # max. iterations for one time step
    'ksp_type': 'preonly',                                       # use preconditioning           
    'pc_type': 'lu',                                             # use LU for precon
    'pc_factor_mat_solver_type': 'mumps'                         # direct solver
})

# output files
out_v = VTKFile("results/velocity.pvd")              # result of velocity
out_p = VTKFile("results/pressure.pvd")              # result of pressure

# set initial values
t = 0.0                                              # initial time
drag_list = []                                       # values for drag force coefficient
lift_list = []                                       # values for lift force coefficient                      

# solve for all times
while t < t_end:
    if COMM_WORLD.rank == 0:                         # for the main process (for MPI):
        print(f"t = {t:.2f}")                        # print time step

    # updates in time
    w0.assign(w)                                     # update functions
    solver.solve()                                   # update solution
    t += dt                                          # update time
    
    # physical attributes v, p, D, T, F
    _v, _p = w.subfunctions                          # get raw data for v, p (split is used as symbolic form)
    D_tensor = sym(grad(_v))                         # get raw data for D
    T = -_p * I + 2 * nu * D_tensor                  # get raw data for stress tensor
    force = dot(T, n)                                # calculate surface force

    # drag and lift force coefficients at cylinder (boundary 9)
    d = assemble(-(2.0 * force[0] / (U_mean * U_mean * L)) * ds(9))         # drag force coefficient
    l = assemble(-(2.0 * force[1] / (U_mean * U_mean * L)) * ds(9))         # lift force coefficient
    drag_list.append((t, d))                                                # update drag force coefficient for the time step
    lift_list.append((t, l))                                                # update lift force coefficient for the time step
    
    out_v.write(_v, time=t)                          # update velocity
    out_p.write(_p, time=t)                          # update pressure
    
print("Finished computations")

# save draf and lift
if COMM_WORLD.rank == 0:
    data_to_save = np.column_stack((
        [item[0] for item in drag_list],             # time
        [item[1] for item in drag_list],             # drag 
        [item[1] for item in lift_list]              # lift
    ))
    
    # Uložíme jako textový soubor s hlavičkou
    np.savetxt('force_data.txt', data_to_save, 
               header='time drag lift', 
               comments='')
    print("Drag/lift saved to force_data.txt")
