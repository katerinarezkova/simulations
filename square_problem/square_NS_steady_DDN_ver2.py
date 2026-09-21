from firedrake import *
import numpy as np

# Mesh
mesh = UnitSquareMesh(40, 40)

# Boundary markers
# left = 1, right = 2, bottom = 3, top = 4
inflow_id = 1
outflow_id = 2
walls = [3, 4]

# Function spaces (Taylor-Hood)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Trial / test
w = Function(W)
v, p = split(w)
v_, p_ = TestFunctions(W)

# Parameters
nu = Constant(0.01)

# Geometry
n = FacetNormal(mesh)

# Forcing (from paper)
x, y = SpatialCoordinate(mesh)
f = as_vector([sin(x) + sin(y), 0])

# ------------------------
# Boundary conditions
# ------------------------

bcs = [
    DirichletBC(W.sub(0), Constant((0, 0)), walls + [inflow_id])
]

# ------------------------
# Negative part function
# ------------------------

def neg_part(s):
    return 0.5*(s - abs(s))

# ------------------------
# Weak form
# ------------------------

conv = inner(dot(v, nabla_grad(v)), v_)*dx
visc = nu*inner(grad(v), grad(v_))*dx
press = - p*div(v_)*dx + p_*div(v)*dx
rhs = inner(f, v_)*dx

# DDN boundary term
vn = dot(v, n)
ddn = -0.5 * neg_part(vn) * inner(v, v_) * ds(outflow_id)

F = conv + visc + press + ddn - rhs

# Jacobian
J = derivative(F, w)

# Solver
problem = NonlinearVariationalProblem(F, w, bcs, J)

solver = NonlinearVariationalSolver(
    problem,
    solver_parameters={
        "snes_type": "newtonls",
        "snes_rtol": 1e-8,
        "ksp_type": "preonly",
        "pc_type": "lu"
    }
)

solver.solve()

# Split solution
v_sol, p_sol = w.split()

# Save
File("velocity.pvd").write(v_sol)
File("pressure.pvd").write(p_sol)

# Diagnostics
div_error = sqrt(assemble(div(v_sol)**2 * dx))
print("div error:", div_error)