from firedrake import *
from firedrake.output import VTKFile
import numpy as np

# --------------------------------
# on a suqare (0,1)x(0,1) mesh
# 2D Heat equation with time-periodic forcing
# period Tperiod = 1
# --------------------------------

# time parameters
Tperiod = 1.0;        
dt = 0.01;            
tmax = 10*Tperiod;    
omega = 2*np.pi/Tperiod; 
n_steps = int(tmax / dt)

# mesh
mesh = Mesh("square.msh")                    # upload file, .msh created using Gmsh
x, y = SpatialCoordinate(mesh) # set x, y as coordinates

# RHS
t = Constant(0.0)
f = sin(omega*t)*exp(-20*((x-0.5)**2+(y-0.5)**2))

# function spaces
V = FunctionSpace(mesh, "CG", 1) 
u = TrialFunction(V)
v = TestFunction(V)

u_sol = Function(V, name="u")
u_prev = Function(V)

a = (u * v / dt + inner(grad(u), grad(v))) * dx
L = (u_prev * v / dt + f * v) * dx

# Dirichlet on boundary
bc = DirichletBC(V, 0.0, "on_boundary")

# ----- Setup for comparison and Output
u_period_start = Function(V)
stored = False
outfile = VTKFile("heat_solution.pvd")

for k in range(n_steps):
    t.assign(float(t) + dt) # Update time constant
    
    # solver
    solve(a == L, u_sol, bcs=bc)
    
    # Update old solution
    u_prev.assign(u_sol)
    
    # Store solution one period before the end (to check periodicity)
    if not stored and float(t) > (tmax - Tperiod):
        u_period_start.assign(u_sol)
        stored = True
        
    # Save to ParaView every 10 steps
    if k % 10 == 0:
        outfile.write(u_sol, time=float(t))
        
# compare solutions
diff = u_sol - u_period_start
l2_err = assemble(dot(diff, diff) * dx)**0.5

print("-" * 35)
print(f"L2 difference after one period = {l2_err:.6e}")
print("-" * 35)