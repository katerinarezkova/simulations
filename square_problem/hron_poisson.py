from firedrake import *

# task
name="task0_2D"

# domain
n=10
mesh = UnitSquareMesh(n, n, "crossed")
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # alternative name FiniteElement("CG",...)
V = FunctionSpace(mesh, P1)
x,y = SpatialCoordinate(mesh)

# the problem
f = 2*(x*(1-x)+y*(1-y))
bc = DirichletBC(V, 0.0, "on_boundary")

# variational formulation
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# init function and solve variational problem
uh = Function(V, name="uh")
solve(a == L, uh, bc)

# save to file
File(f"{name}.pvd").write(uh)
print(f"Saved to {name}.pvd in ParaView.")