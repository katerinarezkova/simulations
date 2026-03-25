nu_input = input("nu = ")
nu = float(nu_input)
r = 0.05                                            # diameter of the object perpendicular to the flow (cylinder) - check from .msh
U = 1.5                                             # maximal velocity at inflow
U_mean = 2/3 * U                                    # mean value of velocity for parabolic flow (= 1)
L = 2 * r                                           # characteristic length of flow (= 0.1)
Re = U_mean * L / nu
print(f"Re = {Re}")