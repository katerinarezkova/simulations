from firedrake import *
from firedrake.output import VTKFile
import numpy as np

# 1. Načtení sítě (předpokládám export z mymesh do formátu, který Firedrake čte, např. msh)
# Pro ukázku použijeme externí mesh nebo vygenerovaný, 
# ale Firedrake preferuje Mesh() objekt.
try:
    # Pokud máte mesh v souboru (např. z Gmsh)
    mesh = Mesh("mesh_cylinder_cor.msh")
except:
    # Náhradní řešení pro demonstraci
    mesh = UnitSquareMesh(32, 32) 

# 2. Definice prostorů (Taylor-Hood: Vector P2 + Scalar P1)
V = VectorFunctionSpace(mesh, "CG", 2)
P = FunctionSpace(mesh, "CG", 1)
W = V * P  # Smíšený prostor ve Firedrake

# 3. Parametry
U = 1.5
nu = Constant(0.001)
dt = 0.1
t_end = 5
theta = Constant(0.5)

# 4. Okrajové podmínky
# Ve Firedrake používáme ID ploch (např. 1, 3, 5) přímo
noslip = Constant((0, 0))
# bndry nahrazeno ID čísel přímo v DirichletBC
bc_walls = DirichletBC(W.sub(0), noslip, 10)
bc_cylinder = DirichletBC(W.sub(0), noslip, 9)

# Inflow - Expression nahrazeno SpatialCoordinate
x, y = SpatialCoordinate(mesh)
v_in = as_vector([U * 4.0 * y * (0.41 - y) / (0.41**2), 0.0])
bc_in = DirichletBC(W.sub(0), v_in, 7)

bcs = [bc_cylinder, bc_walls, bc_in]

# 5. Funkce a formy
n = FacetNormal(mesh)
I = Identity(mesh.geometric_dimension())

w = Function(W)
v, p = split(w) # Aktuální krok
w0 = Function(W)
v0, p0 = split(w0) # Předchozí krok

v_, p_ = TestFunctions(W)

def a(u, vel_conv, v_test):
    D = sym(grad(u))
    # Konvektivní člen + viskózní člen
    return (dot(dot(grad(u), vel_conv), v_test) + inner(2*nu*D, grad(v_test))) * dx

def b(q, u):
    return div(u) * q * dx

# Variační formulace
F1 = a(v, v, v_) - b(p_, v) - b(p, v_)
F0 = a(v0, v0, v_) - b(p_, v0) - b(p, v_) # Tady p zůstává z aktuálního kroku dle vašeho orig. kódu

# Časová derivace + Crank-Nicolson
F = (1.0/dt) * dot(v - v0, v_) * dx + theta * F1 + (1.0 - theta) * F0

# 6. Solver
# Firedrake automaticky počítá derivaci J, pokud ji nepředáte
prob = NonlinearVariationalProblem(F, w, bcs=bcs)
solver = NonlinearVariationalSolver(prob, solver_parameters={
    'snes_type': 'newtonls',
    'snes_atol': 1e-12,
    'snes_rtol': 1e-12,
    'snes_max_it': 20,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps'
})

# 7. Výstup a časová smyčka
out_v = VTKFile("results/velocity.pvd")
out_p = VTKFile("results/pressure.pvd")

t = 0.0
drag_list = []
lift_list = []

while t < t_end:
    if COMM_WORLD.rank == 0:
        print(f"t = {t:.2f}")

    w0.assign(w)
    solver.solve()
    
    t += dt
    
    # Uvnitř while smyčky po solver.solve()
    _v, _p = w.subfunctions  # Správný přístup k pod-funkcím ve Firedrake

    D_tensor = sym(grad(_v))
    # Nezapomeň, že 'n' a 'I' musí být definovány správně před smyčkou
    T = -_p * I + 2 * nu * D_tensor
    force = dot(T, n)

    # Výpočet na hranici válce (ID 5)
    d = assemble(-(2.0 * force[0] / (1.0 * 1.0 * 0.1)) * ds(5))
    l = assemble(-(2.0 * force[1] / (1.0 * 1.0 * 0.1)) * ds(5))
    
    drag_list.append((t, d))
    lift_list.append((t, l))
    
    out_v.write(_v, time=t)
    out_p.write(_p, time=t)