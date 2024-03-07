# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:35:27 2023

@author: Sinurat
"""

# Outer code for setting up the diffusion problem, calculate and plot
import matplotlib.pyplot as plt
import numpy as np
# read in all the schemes, initial conditions and other helper code
from diffusionSchemes import *

def analytic_FTCS():
    """
    Plot the results of simulating heat conduction using the Analytic and FTCS (Forward Time, Central Space) approaches.

    """
    zmin = 0            
    zmax = np.pi
    nz = 50
    K = 0.1           
    dt = 0.01    
    nt = 25000
    
    # Other derived parameters  
    endTime = dt * nt          
    z = np.linspace(zmin, zmax, nz)   
    dz = (zmax - zmin) / (nz - 1)
    d = K * dt / dz**2             
    print('dz =', dz, 'dt =', dt, '\nnon-dimensional diffusion coeficient =', d)
    
    T_analytic = np.exp(-K * endTime) * np.sin(z)
    plt.plot(z, T_analytic, 'r-', label='Analytic')
    
    T = np.sin(z)
    T_FTCS_approx = FTCS_fixed_approximation(T.copy(), K, dz, dt, nt)   
    plt.plot(z, T_FTCS_approx, '+-', label='FTCS')
    plt.legend()
    plt.savefig('4AnalyticSolution_DifferentBoundary.png', bbox_inches='tight')
    plt.show()
    return

analytic_FTCS()


def main():
    """
    Simulate heat conduction in a rod subjected to a time-varying heat source 
    and plot the temperature change.

    """
    zmin = 0
    zmax = 1000
    nz = 20
    K = 1
    Q = -1.5/86400
    endtime = 1e6
    dt = 1e5
    L = zmax - zmin
    T_zmin = 293
    T_zmax = 293    
    order = 10
    t = np.arange(dt, endtime+dt, dt)

    # Main thing below.
    c1 = (Q/(2*K) * (zmax**2 - zmin**2) + T_zmax - T_zmin)/(zmax - zmin)
    c2 = T_zmin + Q/(2*K) * zmin**2 -c1*zmin

    X = np.linspace(zmin, zmax, nz+1)
    Tsteady = -Q/(2*K)*X**2 + c1*X + c2

    
    plt.figure(figsize=(10,10))
    history = []
    for ti in t:
        T = Tsteady.copy()
        for n in range(1, order):
            T += (np.exp(-(n * np.pi/L)**2 * K * ti) * np.sin(n*np.pi*X/L))

        plt.plot(X, T, '-o', label=f"t={ti}")
        history.append(T)
    
    plt.grid(which='both')
    plt.plot(X, Tsteady, 'k--', label="Analytical")
    plt.legend(loc='best')
    plt.savefig('4AnalyticSolution_SteadyState.png', bbox_inches='tight')
    plt.show()
    
main()