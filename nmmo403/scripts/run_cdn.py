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
n_periods = 20            
tmax = n_periods
n_steps = int(tmax / dt)

t = Constant(0.0)

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
out_dir = Path("comparison/CDN")
out_dir.mkdir(parents=True, exist_ok=True)
out = VTKFile(str(out_dir / "solution.pvd"))
v_sol, p_sol = w.subfunctions
v_sol.rename("Velocity")
p_sol.rename("Pressure")

# Record diagnostics at every step
fields = ["step", "time", "kinetic_energy", "grad_norm", "backflow",
          "backflow_energy", "outflow_energy", "inlet_flux", "outlet_flux",
          "mass_imbalance", "div_norm", "newton_iterations"]
csv_file = None
writer = None
if COMM_WORLD.rank == 0:
    csv_file = (out_dir / "diagnostics.csv").open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

try:
    for step in range(n_steps):
        time = (step + 1) * dt
        t.assign(time)
        w.assign(w0)                   
        try:
            solver.solve()
        except Exception as exc:
            if COMM_WORLD.rank == 0:
                (out_dir / "run_status.txt").write_text(
                    f"status=solver_failed\nfailed_step={step + 1}\n"
                    f"failed_time={time:.8f}\nlast_successful_step={step}\n"
                    f"error={type(exc).__name__}: {exc}\n"
                )
                print(f"CDN: solver failed at step {step + 1}, t={time:.3f}: {exc}", flush=True)
            break

        # Integral diagnostics - after a succesful solve
        vn = dot(v_sol, n)
        vn_minus = conditional(vn < 0.0, vn, 0.0)
        vn_plus = conditional(vn > 0.0, vn, 0.0)
        speed2 = inner(v_sol, v_sol)
        inlet_flux = float(assemble(vn * ds(1)))
        outlet_flux = float(assemble(vn * ds(GAMMA_N_MARKER)))
        row = {
            "step": step + 1,
            "time": time,
            "kinetic_energy": float(assemble(0.5 * speed2 * dx)),
            "grad_norm": float(assemble(inner(grad(v_sol), grad(v_sol)) * dx))**0.5,
            "backflow": float(assemble(-vn_minus * ds(GAMMA_N_MARKER))),
            "backflow_energy": float(assemble(-0.5 * vn_minus * speed2 * ds(GAMMA_N_MARKER))),
            "outflow_energy": float(assemble(0.5 * vn_plus * speed2 * ds(GAMMA_N_MARKER))),
            "inlet_flux": inlet_flux,
            "outlet_flux": outlet_flux,
            "mass_imbalance": inlet_flux + outlet_flux,
            "div_norm": float(assemble(div(v_sol)**2 * dx))**0.5,
            "newton_iterations": solver.snes.getIterationNumber(),
        }
        if COMM_WORLD.rank == 0:
            writer.writerow(row)
            csv_file.flush() 
            if step % 10 == 0:
                print(f"CDN: step={step + 1}, t={time:.2f}, "
                      f"E={row['kinetic_energy']:.5g}, "
                      f"backflow={row['backflow']:.5g}, "
                      f"Newton={row['newton_iterations']}", flush=True)

        if step % 10 == 0:     
            out.write(v_sol, p_sol, time=time)
        w0.assign(w)
    else:
        if COMM_WORLD.rank == 0:
            (out_dir / "run_status.txt").write_text(
                f"status=completed\nlast_successful_step={n_steps}\nlast_time={tmax:.8f}\n"
            )
            print("CDN: finished all time steps.")
finally:
    if csv_file is not None:
        csv_file.close()
