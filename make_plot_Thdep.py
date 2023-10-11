import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os

# positive convention for modulus and compliance
# J* = J1 + iJ2
# M* = M1 + iM2
# M* = 1 / J* -> M1 = J1/|J|^2 and M2 = -J2/|J|^2

def normcompliance_YT16(tau_n,T_h):
    
    A_b = 0.664
    alpha = 0.38
    tau_p = 6e-05
    beta = 0

    if T_h < 0.91:
        A_p = 0.01
    elif T_h < 0.96:
        A_p = 0.01 + (0.4 * (T_h - 0.91))
    elif T_h < 1:
        A_p = 0.03
    else:
        A_p = 0.03 + beta

    if T_h < 0.92:
        sigma_p = 4
    elif T_h < 1:
        sigma_p = 4 + (37.5 * (T_h - 0.92))
    else:
        sigma_p = 7

    J1_norm = 1 + ((A_b*(tau_n**alpha))/alpha) + (((np.sqrt(2*np.pi))/2)*A_p*sigma_p*(1-sp.special.erf((np.log(tau_p/tau_n))/(np.sqrt(2)*sigma_p))))
    J2_norm = (np.pi/2)*((A_b*(tau_n**alpha))+(A_p*np.exp(-((np.log(tau_p/tau_n))**2)/(2*(sigma_p**2))))) + tau_n

    return J1_norm, J2_norm

def compliance2modulus(J1,J2):

    J_mod = J1**2 + J2**2
    M1 = J1 / J_mod
    M2 = J2 / J_mod

    return M1, M2

def normmod_Mx(tau_n):

    M1_norm = 1 / (tau_n**2 + 1)
    M2_norm = tau_n / (tau_n**2 + 1)

    return M1_norm, M2_norm

def modulus_delta(M1,M2,M1_Mx,M2_Mx):

    return M1-M1_Mx, M2-M2_Mx

n_tau = 100
n_theta = 3
tau = 10**np.linspace(-12,3,n_tau)
arr_theta = np.linspace(0.8,1.1,n_theta)

M1_Mx,M2_Mx = normmod_Mx(tau)
J1 = np.zeros((n_theta,n_tau))
J2 = J1.copy()
M1 = J1.copy()
M2 = J1.copy()
M1_delta = J1.copy()
M2_delta = J1.copy()

for i in range(n_theta):

    theta = arr_theta[i]
    J1[i,:],J2[i,:] = normcompliance_YT16(tau,theta)
    M1[i,:],M2[i,:] = compliance2modulus(J1[i,:],J2[i,:])
    M1_delta[i,:],M2_delta[i,:] = modulus_delta(M1[i,:],M2[i,:],M1_Mx,M2_Mx)

n_rows = 2
n_cols = 2
fig,ax=plt.subplots(n_rows,n_cols,figsize=(6*n_cols,3*n_rows))
colors = ['red','green','blue','pink','cyan','magenta','yellow']
arr_colors = colors * 10

for i in range(n_theta):    

    ax[0,0].plot(tau,M1[i],color=arr_colors[i],label=f'Th={arr_theta[i]:.2f}')
    ax[0,1].plot(tau,M2[i],color=arr_colors[i])
    ax[1,0].plot(tau,M1_delta[i],arr_colors[i])
    ax[1,1].plot(tau,M2_delta[i],arr_colors[i])

ax[0,0].set_xscale('log', base=10)
ax[0,0].set_xlabel(r'Normalised timescale $\tau/\tau_M$')
ax[0,0].set_ylabel(r'$M_1/M_U$')
ax[0,0].invert_xaxis()
ax[0,1].set_xscale('log', base=10)
ax[0,1].set_xlabel(r'Normalised timescale $\tau/\tau_M$')
ax[0,1].set_ylabel(r'$M_2/M_U$')
ax[0,1].invert_xaxis()

ax[1,0].set_xscale('log', base=10)
ax[1,0].set_xlabel(r'Normalised timescale $\tau/\tau_M$')
ax[1,0].set_ylabel(r'$(M_1-M_1^{Mx})/M_U$')
ax[1,0].invert_xaxis()
ax[1,1].set_xscale('log', base=10)
ax[1,1].set_xlabel(r'Normalised timescale $\tau/\tau_M$')
ax[1,1].set_ylabel(r'$(M_2-M_2^{Mx})/M_U$')
ax[1,1].invert_xaxis()

ax[0,0].legend()
plt.tight_layout()
plt.savefig(os.path.join("plot_output_Thdep","M_YT16_Thdep.jpg"),dpi=1200)
plt.close()
