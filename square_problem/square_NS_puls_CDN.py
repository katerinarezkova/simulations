from firedrake import *
from firedrake.output import VTKFile
import numpy as np

## ---------------------------------------
# Pulsatile flow with CDN (classical do-nothing) boundary condition
# Ω = (0,1)^2, ΓN = {0} x (0,1) (do-nothing, natural -> NO bc there)
# ΓD = boundary \ ΓN (homogeneous Dirichlet, v = 0)
# f(x,y,t) = sin(2*pi*t) * (sin(x) + sin(y), 0)
# time-periodic problem: v(0,.) = v(T,.), T = 1
# --------------------------------------

# mesh -- built-in unit square, no external mesh file / Gmsh needed.
# Firedrake auto-tags the four sides: 1 = x=0, 2 = x=1, 3 = y=0, 4 = y=1
nx, ny = 20, 20                              # cells per side; increase for finer resolution
mesh = UnitSquareMesh(nx, ny)

# finite element space (Taylor-Hood: P2 + P1)
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
W = V * P
x, y = SpatialCoordinate(mesh)

# --------------------------------
# time parameters
# --------------------------------
Tperiod = 1.0                        # forcing period (matches PDF: T = 1)
dt = 0.01
n_periods = 10                       # run several periods to reach periodic steady state
tmax = n_periods * Tperiod
n_steps = int(tmax / dt)

t = Constant(0.0)

# the problem
f = as_vector((sin(2*np.pi*t)*(sin(x) + sin(y)), 0))
g = as_vector((0, 0))                       # homogeneous Dirichlet on ΓD
k = 3                                       # 1, 2, 3, 4 -> controls nu, as before
nu = Constant(5 * 10**(-k))

# --------------------------------------------------------------------------
# BOUNDARY MARKERS -- Firedrake's built-in UnitSquareMesh convention:
#   1 = x=0 (left)   -> Gamma_N (do-nothing, NO bc applied -- left out on purpose)
#   2 = x=1 (right)  -> Gamma_D
#   3 = y=0 (bottom) -> Gamma_D
#   4 = y=1 (top)    -> Gamma_D
# --------------------------------------------------------------------------
GAMMA_D_MARKERS = (2, 3, 4)

bc_walls = DirichletBC(W.sub(0), g, GAMMA_D_MARKERS)
bcs = [bc_walls]

# ------------------------------------------------------------------------------------------------------------------
# VARIATIONAL PROBLEM (unsteady)
# ------------------------------------------------------------------------------------------------------------------

w0 = Function(W)                # solution at previous time step
v0, p0 = split(w0)

w = Function(W)                 # unknown at current time step
v, p = split(w)

v_, p_ = TestFunctions(W)

# convective and viscous term (unchanged from steady script)
def a(u, v_conv, phi):
    return (dot(dot(grad(u), v_conv), phi) + inner(nu*grad(u), grad(phi))) * dx

# pressure / incompressibility coupling (unchanged)
def b(q, u):
    return q * div(u) * dx

# backward-Euler time derivative + steady operator, no ΓN boundary term (do-nothing)
F = (inner(v - v0, v_) / dt) * dx + a(v, v, v_) - b(p_, v) - b(p, v_) - inner(f, v_) * dx

# ------------------------------------------------------------------------------------------------------------------
# SOLVER -- built ONCE, reused every time step
# ------------------------------------------------------------------------------------------------------------------
prob = NonlinearVariationalProblem(F, w, bcs=bcs)
solver = NonlinearVariationalSolver(prob, solver_parameters={
    'snes_type': 'newtonls',
    'snes_linesearch_type': 'bt',
    'snes_atol': 1e-12,
    'snes_rtol': 1e-12,
    'snes_max_it': 20,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
})

# output files
out_v = VTKFile(f"results_CDN_pulsatile/v_sqr_pulsatile_CDN_{k}.pvd")
out_p = VTKFile(f"results_CDN_pulsatile/p_sqr_pulsatile_CDN_{k}.pvd")

# ----- Setup for periodicity comparison
w_period_start = Function(W)
stored = False

for step in range(n_steps):
    t.assign(float(t) + dt)

    # use previous solution as initial guess for Newton (warm start)
    w.assign(w0)
    solver.solve()

    # advance
    w0.assign(w)

    # store solution one period before the end, to check periodicity
    if not stored and float(t) > (tmax - Tperiod):
        w_period_start.assign(w)
        stored = True

    # save every 10 steps
    if step % 10 == 0:
        v_sol, p_sol = w.subfunctions
        out_v.write(v_sol, time=float(t))
        out_p.write(p_sol, time=float(t))

# ------------------------------------------------------------------------------------------------------------------
# compare solution one period apart (check convergence to periodic state)
# ------------------------------------------------------------------------------------------------------------------
v_final, p_final = w.subfunctions
v_prev_period, p_prev_period = w_period_start.subfunctions

diff_v = v_final - v_prev_period
l2_err_v = assemble(dot(diff_v, diff_v) * dx)**0.5

if COMM_WORLD.rank == 0:
    print("-" * 45)
    print(f"nu = {float(nu):.3e}  (k = {k})")
    print(f"L2 velocity difference after one period = {l2_err_v:.6e}")
    print("-" * 45)
