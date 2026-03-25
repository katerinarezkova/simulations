Re_input = input("Re = ")
Re = float(Re_input)
r = 0.05                                            # diameter of the object perpendicular to the flow (cylinder) - check from .msh
L = 2 * r                                           # characteristic length of flow (= 0.1)
nu = 0.001
U_mean = nu * Re / L
U = U_mean * 3/2
print(f"U = {U}")