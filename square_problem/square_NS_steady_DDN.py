from firedrake import *
from firedrake.output import VTKFile
import numpy as np

# ---------------------------------------
# here we try to reconstruct the problem solved by Braack
# that is, STEADY flow with DDN
# f(x,y) = (sin(x) + sin(y), 0)
# --------------------------------------

# mesh
# mesh = Mesh("square.msh")                    # upload file, .msh created using Gmsh
# k_max = 3
n_intervals = 64 * 2  
mesh = UnitSquareMesh(n_intervals, n_intervals)

# 1: x == 0 (flow)
# 2: x == 1 
# 3: y == 0 
# 4: y == 1 

# finite element space (Taylor-Hood: P2 + P1)
V = VectorFunctionSpace(mesh, "CG", 2)              # vector space for u
P = FunctionSpace(mesh, "CG", 1)                    # scalar space for p
P_proj = FunctionSpace(mesh, "DG", 0)         # projection space for LPS stabilization
W = V * P                                           # mixed space for (u,p)
x, y = SpatialCoordinate(mesh)                      # set x, y as coordinates

# the problem
# h = as_vector((0, 0))
f = as_vector((sin(x) + sin(y), 0))
# v_r = as_vector([0, 0])
g = as_vector((0, 0))                           # set (0,0) value (later used at u = (0,0) on boundaries)
# k = 3 # 1, 2, 3, 4
Re_input = float(input("Enter Reynolds number Re: "))
Re = Re_input
# nu = Constant(0.5)
nu = Constant(1 / Re)

# boundary conditions
# bc_walls = DirichletBC(W.sub(0), g, 5)        # W.sub(0) gives me velocity (.sub(1) is pressure), (0,0), no. 10 based on the .msh (.geo) file is the walls
# bc_walls = [DirichletBC(W.sub(0), Constant((0, 0)), (2, 3, 4))]
bc2 = DirichletBC(W.sub(0), Constant((0, 0)), 2)
bc3 = DirichletBC(W.sub(0), Constant((0, 0)), 3)
bc4 = DirichletBC(W.sub(0), Constant((0, 0)), 4)

# bcs = [bc_walls]                # put together all boundaries
bcs = [bc2, bc3, bc4]
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

def a(u, phi):                                                                 
    a = (inner(nu*grad(u), grad(phi))) * dx
    return a           

# negative part
def neg(y):
    y_neg = (abs(y) - y) / 2
    return y_neg

# pressure and incompressibility
def b(q, u):                                         # p * div(v_) // p_ * div(v)
    return q * div(u) * dx                           # bilinear form

# convective and viscous term
def a_ns(u, v_conv, phi):
    return (dot(dot(grad(u), v_conv), phi) + inner(nu * grad(u), grad(phi))) * dx

# stabilization LPS for the convective term

def stab_term(u, phi, v_conv):
    h = CellSize(mesh)
    delta = Constant(0.1) * h / sqrt(dot(v_conv, v_conv) + 1e-8)
    # kappa_u = dot(grad(u),u) - project(dot(grad(u),u), P_proj)
    # kappa_phi = dot(grad(phi),u) - project(dot(grad(phi),u), P_proj)
    # return delta * dot(kappa_u, kappa_phi) * dx
    # res = dot(v_conv, grad(u))
    # tau = delta * dot(v_conv, grad(phi))
    # return inner(res, tau) * dx
    res = dot(v_conv, grad(u))
    return delta * inner(res, dot(v_conv, grad(phi))) * dx

# ------------------------------------------------------------------------------------------------------------------
# SOLVE FOR STOKES
# ------------------------------------------------------------------------------------------------------------------

if COMM_WORLD.rank == 0:                         # for the main process (for MPI):
    print(f"Solver for Stokes velocity started.")



F_r = a(v, v_) - b(p_, v) - b(p, v_) - inner(f, v_) * dx

solve(F_r == 0, w, bcs=bcs)
v_r = Function(V).assign(w.sub(0))
w0.assign(w)

if COMM_WORLD.rank == 0:                         # for the main process (for MPI):
    print(f"-- Stokes velocity computed.")
   
# ------------------------------------------------------------------------------------------------------------------
# OSEEN CORRECTION FOR IG
# ------------------------------------------------------------------------------------------------------------------   
   
def a_oseen(u, phi, b):
    return (dot(dot(grad(u), b), phi) + inner(nu * grad(u), grad(phi))) * dx
   
if COMM_WORLD.rank == 0:                         # for the main process (for MPI):
    print(f"Solver for Oseen correction started.")  
   
F_oseen = a_oseen(v, v_, v_r) - b(p_, v) - b(p, v_) - inner(f, v_) * dx
solve(F_oseen == 0, w, bcs=bcs)
v_oseen = Function(V).assign(w.sub(0))
w0.assign(w)

if COMM_WORLD.rank == 0:                         # for the main process (for MPI):
    print(f"-- Oseen correction computed.")

# ------------------------------------------------------------------------------------------------------------------
# SOLVER FOR DDN
# ------------------------------------------------------------------------------------------------------------------

# F = a_ns(v, v, v_) - b(p_, v) - b(p, v_) - inner(f, v_) * dx - 0.5 * neg(dot(v, n)) * dot(v - v_r, v_) * ds(1) + stab_term(v, v_)
# F = a_ns(v, v, v_) - b(p_, v) - b(p, v_) - inner(f, v_) * dx - 0.5 * neg(dot(v, n)) * dot(v - v_r, v_) * ds(1)

if COMM_WORLD.rank == 0:                         # for the main process (for MPI):
    print(f"Solver for the whole NS started.") 

F = (a_ns(v, v_oseen, v_) - b(p_, v) - b(p, v_) - inner(f, v_) * dx - 0.5 * neg(dot(v_oseen, n)) * dot(v - v_oseen, v_) * ds(1) + stab_term(v, v_, v_oseen) )


# F = (
#     inner(dot(v, nabla_grad(v)), v_) * dx
#     + nu * inner(grad(v), grad(v_)) * dx
#     - p * div(v_) * dx
#     + p_ * div(v) * dx
#     - inner(f, v_) * dx
#     - 0.5 * neg(dot(v, n)) * dot(v, v_) * ds(1)
# )   
   
    
# parameters for solver

lu = {
    "snes_monitor": "",
    "snes_type": "newtonls",
    "snes_max_it": 50,
    "snes_rtol": 1e-6,
    "snes_atol": 1e-6,
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}


# problem = fd.NonlinearVariationalProblem(F,w,bcs,J)
prob = NonlinearVariationalProblem(F, w, bcs=bcs)                # (weak formulation, weak solution, boundary conditions)
solver = NonlinearVariationalSolver(prob, solver_parameters=lu)

# # CYKLUS CONTINUATION
# for k_val in range(1, k_max + 1):
#     current_nu = 5 * 10**(-k_val)
#     nu.assign(current_nu)
    
#     if COMM_WORLD.rank == 0:
#         print(f"\n>>> Computing for k={k_val}, (nu={current_nu:g})")
    
#     solver.solve()
#     w0.assign(w)

solver.solve()

v_sol, p_sol = w.subfunctions
v_sol.rename("velocity")
p_sol.rename("pressure")

out_v = VTKFile(f"results_DDN/v_sqr_steady_DDN_{Re}.pvd")              # result of velocity
out_p = VTKFile(f"results_DDN/p_sqr_steady_DDN_{Re}.pvd")              # result of pressure

out_v.write(v_sol)
out_p.write(p_sol)

if COMM_WORLD.rank == 0:                         # for the main process (for MPI):
    print(f"Finished computations.") 
    print(f"- saved to results_DDN/v_sqr_steady_DDN_{Re}.pvd")
    print(f"- saved to results_DDN/p_sqr_steady_DDN_{Re}.pvd")