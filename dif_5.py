# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:35:27 2023

@author: Sinurat
"""
# Outer code for setting up the diffusion problem, calculate and plot
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
# read in all the schemes, initial conditions and other helper code
from diffusionSchemes import *


def main():
    """
    Determine the L2 norm error using linear regession between the approximate and analytical solution

    """
    # Parameters
    zmin = 0          
    zmax = 100 
    endTime = 3     
    K = 1       
    d = 0.4
    
    # Create an array
    nz = np.arange(5, 50, 40)
    
    # List to store the result
    dt_list = []
    dz_list = []
    L2_list = []
    error_dz_list = []
    error_dt_list = []
   
    for i in nz:
        dz = (zmax - zmin) / (i - 1)
        dt = dz**2 * d/ K
        nt = int(endTime / dt)
        T_analytic = np.exp(-K * endTime) * np.sin(dz)
        T = np.sin(nz)
        T_approx = FTCS_fixed_approximation(T.copy(), K, dz, dt, nt)
        dt_list.append(dt)
        dz_list.append(dz)
        L2 = error_L2(dz, T_approx, T_analytic)
        L2_list.append(L2)
        order_accuracy = error_L2(dz, T_approx, T_analytic)
        error_dz_list.append(order_accuracy)
        error_dt_list.append(order_accuracy)
    
    # Plot L2 error
    plt.loglog(dz_list, L2_list, label=f'L2 error')
    plt.legend()
    plt.xlabel("dx")
    plt.ylabel("L2 Norm")
    plt.savefig('5Error_Norm(L2).png', bbox_inches='tight')
    plt.show()
    
    # Linear regression to determine convergence
    slope_dz = linregress(np.log(dz_list), np.log(error_dz_list))
    slope_dt = linregress(np.log(dt_list), np.log(error_dt_list))
      
    convergence_n_dz = round(slope_dz.slope, 2)
    convergence_n_dt = round(slope_dt.slope, 2)
    
    # Plot convergence
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    plt.rcParams["font.size"] = 12
    axs[0].loglog(np.log(dz_list), np.log(error_dz_list), marker='*', label=f'n={convergence_n_dz}')
    axs[0].set_xlabel('dz')
    axs[0].set_ylabel('error')
    axs[0].set_title('dz convergence')
    
    axs[1].loglog(np.log(dt_list), np.log(error_dt_list), marker='*', label=f'n={convergence_n_dt}')
    axs[1].set_xlabel('dt')
    axs[1].set_ylabel('error')
    axs[1].set_title('dt convergence')
    
    for i in axs:
        i.set_ylim([0.1,12])
        i.legend(loc='best')
        i.grid(which='both') 
    #plt.ylim([0.1,12])
   
    fig.tight_layout()
    plt.savefig('5Order_Convergence.png', bbox_inches='tight')
    plt.show()
    
main()