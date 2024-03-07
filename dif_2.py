# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 19:08:16 2023

@author: Sinurat
"""
# Outer code for setting up the diffusion problem, calculate and plot
import matplotlib.pyplot as plt
import numpy as np
# read in all the schemes, initial conditions and other helper code
from diffusionSchemes import *

def main():
    """
    Simulates, calculates, and visualizes a diffusion problem using FTCS and 
    BTCS schemes with varying time steps.
    The main function sets up the parameters, initializes the grid, 
    and computes temperature profiles using FTCS and BTCS schemes. 
    It then creates two plots: one showing temperature profiles at different 
    time steps, and the other showing the difference between FTCS and BTCS solutions.

    """
    # Parameters
    zmin = 0                    # Start of model domain (m)                             
    zmax = 1e3                  # End of model doman (m)                               
    nz = 21                     # Number of grid points, including both ends    
    d = 0.504                   # Diffusion
    K = 1.                      # The diffusion coefficient (m^2/s)
    Tinit = 293.                # The initial condition
    Q = -1.5/86400              # The heating rate
    
    # Other derived parameters
    dz = (zmax - zmin)/(nz - 1) # The Grid Spacing
    dt = d * dz**2 / K          # The time step
    print('dx =', dz, 'dt =', dt, '\non-dimensional diffusion coeficient =', d)
    
    # Height points
    z = np.linspace(zmin, zmax, nz)
    
    # Initial condition
    T = Tinit * np.ones(nz)
    nt_values = [0, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    
    # Plot the solutions Figure 1
    fig, axs = plt.subplots(3, 3, figsize = (24,18))
    plt.rcParams["font.size"] = 12

    for i in range(len(nt_values)):
        # Diffusion using FTCS and BTCS
        T_FTCS = FTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt_values[i]) 
        T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt_values[i]) 
        x, y = i // 3, i % 3
        axs[x, y].plot(T_FTCS - Tinit, z, 'r--', linewidth=1, label='FTCS', marker='^', markersize=4)
        axs[x, y].plot(T_BTCS - Tinit, z, 'c:', linewidth=1, label='BTCS', marker='o', markersize=4)
        axs[x, y].set_title(f'n_t = {nt_values[i]}')

        axs[x, y].set_ylim(0, 1000)
        axs[x, y].legend(loc='best')
        axs[x, y].set_xlabel('$T - T_0$ (K)')
        axs[x, y].set_ylabel('z (m)')
        axs[x, y].grid(which='both')
    fig.tight_layout()
    plt.savefig('2Unstable.png', bbox_inches='tight')
    plt.show()

    # Plot the solutions Figure 1
    for i in range(len(nt_values)):
        T_FTCS = FTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt_values[i]) 
        T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt_values[i]) 
        x, y = i // 3, i % 3
        axs[x, y].plot(T_FTCS - T_BTCS, z, 'r--', linewidth=1, label='FTCS - BTCS', marker='^', markersize=4)
        axs[x, y].set_title(f'd_t = {nt_values[i]}')
        axs[x, y].set_ylim(0, 1000)
        axs[x, y].legend(loc='best')
        axs[x, y].set_xlabel('$T - T_0$ (K)')
        axs[x, y].set_ylabel('z (m)')
        axs[x, y].grid(which='both')
    fig.tight_layout()
    plt.savefig('2Unstable_Diff.png', bbox_inches='tight')
    plt.show()
    
main()

def BTCS_one_step():
    z_min = 0.              
    z_max = 1e3
    K = 1
    T_init = 293.
    Q = -1.5 / 86400
    nz = 21                       
    dz = (z_max - z_min) / (nz - 1)  
    dt = [1e8,1e7,1e6]  
    endTime = 1e8

    plt.figure()
    z = np.linspace(z_min, z_max, nz)
    T = T_init * np.ones(nz)

    for i in dt:
        nt = int(endTime / i)
        d = K * i / dz**2
        print("dz  =", dz, "dt =", i, "non-dimensional diffusion coefficient =", d)
        
        T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, i, nt)
        
        plt.plot(T_BTCS - T_init, z, label=f'nt = {nt}, dt = {i}', linestyle='dashed')

    plt.xlabel('$T-T_0$ (K)')
    plt.ylabel('z (m)')
    plt.legend()
    plt.title('BTCS in Difference Step')
    plt.savefig('2BTCS_DiffStep.png', bbox_inches='tight')
    plt.tight_layout()

BTCS_one_step()


