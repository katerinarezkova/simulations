import numpy as np
import matplotlib.pyplot as plt

try:
    data = np.loadtxt('force_data.txt', skiprows=1)
    time = data[:, 0]
    drag = data[:, 1]
    lift = data[:, 2]
    
    Re = 100 

    plt.figure(figsize=(10, 6))

    plt.plot(time, drag, 'b-', label=f'Drag coefficient $C_D$')
    plt.plot(time, lift, 'r-', label=f'Lift coefficient $C_L$')
    
    plt.title(f'Flow around cylinder ($Re={Re}$)')
    plt.xlabel('Time [s]')
    plt.ylabel('Coefficients lift/drag')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    
    plt.savefig('final_plot.pdf', bbox_inches='tight')
    print("Graph saved to final_plot.pdf.")
    
except FileNotFoundError:
    print("Error: no force_data.txt file.")