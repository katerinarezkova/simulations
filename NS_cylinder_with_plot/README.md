# 2D Navier-Stokes Simulation: Flow Around a Cylinder

This project implements a numerical simulation of the non-stationary incompressible Navier-Stokes equations using the **Firedrake** finite element library. The setup follows the well-known **DFG 2D-2 (Re=100)** benchmark, which characterizes the periodic vortex shedding behind a circular cylinder (Von Kármán vortex street).

## 1. Physical and Mathematical Model

The fluid flow is governed by the conservation of momentum and mass (continuity equation) for a velocity field $\mathbf{u}$ and pressure $p$:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} - \nu \Delta \mathbf{u} + \nabla p = 0$$
$$\nabla \cdot \mathbf{u} = 0$$

### Benchmark Parameters:
* **Kinematic Viscosity:** $\nu = 0.001$
* **Reynolds Number:** $Re = 100$ (based on $U_{mean} = 1.0$ and cylinder diameter $L = 0.1$)
* **Inflow Profile:** Parabolic with a maximum velocity $U_{max} = 1.5$
* **Geometry:** A rectangular channel $[0, 2.2] \times [0, 0.41]$ with a circular obstacle centered at $(0.2, 0.2)$.

## 2. Numerical Implementation

* **Spatial Discretization:** Finite Element Method (FEM) using Taylor-Hood elements ($P_2$ for velocity, $P_1$ for pressure) to ensure LBB stability.
* **Time Discretization:** $\theta$-method (Crank-Nicolson scheme, $\theta = 0.5$) for second-order temporal accuracy.
* **Nonlinear Solver:** Newton method with line search (`newtonls`).
* **Linear Solver:** **MUMPS** direct solver (LU decomposition) via PETSc for robust handling of the saddle-point system.

## 3. Project Structure

| File | Purpose |
| :--- | :--- |
| `ns_cylinder_fire.py` | Main Firedrake script. Runs the solver, exports numerical data and generates PDF plots. |
| `results/` | Directory containing `.pvd` files for visualization in **ParaView**. |

## 4. Instructions for Use

### Running the Simulation
It is recommended to run the simulation in parallel to speed up the LU factorization. 

`srun -p express3 -n 4 -u python simulation.py`

## 5. Outputs and Validation

The primary outputs are the dimensionless Drag ($C_D$) and Lift ($C_L$) coefficients, calculated by integrating the total stress tensor over the cylinder surface $S$:

$$C_D = \frac{2 F_{drag}}{\rho U_{mean}^2 L}, \quad C_L = \frac{2 F_{lift}}{\rho U_{mean}^2 L}$$

For $Re=100$, the flow exhibits periodic vortex shedding:
* **Expected $C_D$:** Should oscillate around a mean value of approximately **3.22**.
* **Expected $C_L$:** Should oscillate symmetrically between approximately **[-1.0, 1.0]**.

---