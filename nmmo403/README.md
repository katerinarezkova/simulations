# Classical vs Directional Do-Nothing Condition 

This project compares the classical do-nothing (CDN) and directional do-nothing (DDN) boundary conditions for unsteady incompressible Navier–Stokes flow in a two-dimensional square domain with a prescribed parabolic inlet profile.

The simulations were implemented in **Firedrake**, with results visualized in **ParaView**. The comparison focuses on kinetic energy, backflow through the outlet boundary, and numerical solver convergence.

## Project Structure

- **[Report](report.pdf)** – Mathematical formulation, numerical setup, results, and discussion.
- **[Scripts](scripts/)** – Firedrake simulations for CDN and DDN, together with a Python script for comparing the results.
- **[Results](results/)** – Comparison plots and animations of the simulated flow.

## Main Observation

For the investigated configuration, the CDN simulation failed to converge at approximately \(t=6.41\), whereas the DDN simulation successfully reached the final time \(t=20\).

The results illustrate differences in numerical robustness and energy behavior associated with the two boundary conditions. Further details can be found in the report.

## Reference

Braack, M., & Mucha, P. B. (2014). *Directional Do-Nothing Condition for the Navier–Stokes Equations*. Journal of Computational Mathematics, 32(5), 507–521.
