from firedrake import *
from firedrake.output import VTKFile
import csv
from pathlib import Path

# Mesh
nx, ny = 20, 20
mesh = UnitSquareMesh(nx, ny)

# FEM space
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
W = V * P
x, y = SpatialCoordinate(mesh)
n = FacetNormal(mesh)

# Time parameters (steady)
dt = 0.01
tmax = 20          
n_steps = int(tmax / dt)

# Viskosity
k = int(input("Choose k for viscosity (nu = 10^-k), preferably 1, 2, 3, or 4: "))
nu = Constant(1 * 10**(-k))

# Boundary markers: 1 = inflow (left), 2 = natural outflow (right),
# 3 and 4 = no-slip walls (bottom and top)
GAMMA_D_MARKERS = (3, 4)
GAMMA_N_MARKER = 2

U_max = Constant(1.0) 
y_min = 0.1
y_max = 0.4
width = y_max - y_min

parabola = U_max * (4.0 / width**2) * (y - y_min) * (y_max - y)

u_x = conditional(y >= y_min, conditional(y <= y_max, parabola, 0.0), 0.0)

vin= as_vector([u_x, 0.0])

bc_walls = DirichletBC(W.sub(0), as_vector([0,0]), GAMMA_D_MARKERS)
bc_in = DirichletBC(W.sub(0), vin, 1)
bcs = [bc_walls, bc_in]

# Navier-Stokes variational formulation
w0 = Function(W)
v0, p0 = split(w0)

w = Function(W)
v, p = split(w)
v_, p_ = TestFunctions(W)

def a(u, v_conv, phi):
    return (dot(dot(grad(u), v_conv), phi) + inner(nu * grad(u), grad(phi))) * dx

def b(q, u):
    return q * div(u) * dx

F = inner(v - v0, v_) / dt * dx + a(v, v, v_) - b(p_, v) - b(p, v_)

v_n = dot(v, n)
v_n_minus = conditional(v_n < 0.0, v_n, 0.0)
F += -0.5 * v_n_minus * dot(v, v_) * ds(GAMMA_N_MARKER)

# Solver
prob = NonlinearVariationalProblem(F, w, bcs=bcs)
solver = NonlinearVariationalSolver(prob, solver_parameters={
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_atol": 1e-12,
    "snes_rtol": 1e-12, 
    "snes_max_it": 20,
    "ksp_type": "preonly", 
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
})

# Output files
out_dir = Path("comparison/DDN")
out_dir.mkdir(parents=True, exist_ok=True)

out = VTKFile(str(out_dir / "solution.pvd"))

v_sol, p_sol = w.subfunctions
v_sol.rename("Velocity")
p_sol.rename("Pressure")

# CSV diagnostics
fields = [
    "step",
    "time",
    "kinetic_energy",
    "backflow_energy",
    "backflow"
]

csv_file = None
writer = None

if COMM_WORLD.rank == 0:
    csv_file = (out_dir / "diagnostics.csv").open(
        "w", newline=""
    )
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

# Time loop
try:
    for step in range(n_steps):

        time = (step + 1) * dt
        w.assign(w0)

        try:
            solver.solve()

        except Exception as exc:
            if COMM_WORLD.rank == 0:
                print(
                    f"DDN failed at step {step + 1}, "
                    f"t={time:.3f}: {exc}",
                    flush=True
                )
            break

        # Diagnostics
        vn = dot(v_sol, n)
        vn_minus = conditional(vn < 0.0, vn, 0.0)
        speed2 = inner(v_sol, v_sol)

        row = {
            "step": step + 1,
            "time": time,

            "kinetic_energy": float(
                assemble(0.5 * speed2 * dx)
            ),

            "backflow_energy": float(
                assemble(
                    -0.5 * vn_minus * speed2
                    * ds(GAMMA_N_MARKER)
                )
            ),

            "backflow": float(
                assemble(
                    -vn_minus * ds(GAMMA_N_MARKER)
                )
            )
        }

        # Save diagnostics
        if COMM_WORLD.rank == 0:
            writer.writerow(row)
            csv_file.flush()

            if step % 10 == 0:
                print(
                    f"DDN: step={step + 1}, "
                    f"t={time:.2f}, "
                    f"E={row['kinetic_energy']:.5g}, "
                    f"backflow={row['backflow']:.5g}",
                    flush=True
                )

        # Save results for ParaView
        if step % 10 == 0:
            out.write(v_sol, p_sol, time=time)

        # Advance solution
        w0.assign(w)

    else:
        if COMM_WORLD.rank == 0:
            print("DDN: finished all time steps.")

finally:
    if csv_file is not None:
        csv_file.close()
    if csv_file is not None:
        csv_file.close()
