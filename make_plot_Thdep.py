import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import os
import pandas as pd
import sys
sys.path
sys.executable
import pyvisco as visco
from pyvisco import styles
styles.format_fig()

# positive convention for compliance
# J* = J1 + iJ2
# M* = M1 - iM2
# M* = 1 / J* -> M1 = J1/|J|^2 and M2 = J2/|J|^2

f_out = os.path.join(".","plot_output_Thdep")
f_plot = os.path.join(".","plot_output_Thdep")
os.makedirs(f_out,exist_ok=True)
os.makedirs(f_plot,exist_ok=True)

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
    J2_norm = (np.pi/2)*((A_b*(tau_n**alpha)) + (A_p*np.exp(-((np.log(tau_p/tau_n))**2)/(2*(sigma_p**2))))) + tau_n

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

def plot_modulus():

    n_tau = 1000
    n_theta = 5
    tau = 10**np.linspace(-11,4,n_tau)
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
    colors = ['red','green','blue','orange','purple']
    n_multiply = int(n_theta / len(colors)) + 1
    arr_colors = colors * n_multiply

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

def plot_fit(df_master,df_GMaxw,units,theta):
    
    modul = df_master.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)
    if df_master.domain == 'freq':
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,0.75*4))
        df_master.plot(x='f', y=[stor], label=["{}'(filter)".format(modul)],
            ax=ax1, logx=True, logy=False, color=['C0'], 
            alpha=0.5, ls='', marker='o', markersize=3)
        df_master.plot(x='f', y=[loss], label=["{}''(filter)".format(modul)],
            ax=ax2, logx=True, logy=False, color=['C1'], 
            alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='f', y=[stor], label=["Prony fit"],
            ax=ax1, logx=True, logy=False, ls='-', lw=2, color=['C0'])
        df_GMaxw.plot(x='f', y=[loss], label=["Prony fit"],
            ax=ax2, logx=True, logy=False, ls='-', lw=2, color=['C1'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage modulus ({})'.format(units[stor]))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Loss modulus ({})'.format(units[stor])) 
        ax1.legend()
        ax2.legend()
    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots(figsize=(4,0.75*4))
        df_master.plot(x='t', y=[relax], ax=ax1, logx=True, logy=False, 
            color=['gray'], ls='', marker='o', markersize=3,
            label=["{}(filter)".format(modul)])
        df_GMaxw.plot(x='t', y=[relax], ax=ax1, label=['Prony fit'],
            logx=True, logy=False, ls='-', lw=2, color=['r'])
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units[relax])) 
        ax1.legend()
    plt.savefig(os.path.join(f_plot, f"relaxation_modulus_Th_{theta:.2f}.jpg"), dpi=600)   

def fit_prony(Th,n_prony):

    T_sol = 1326
    T_input = Th*(T_sol + 273.15) - 273.15
    n_tau = 1001
    tau = np.flip(10**np.linspace(-11,4,n_tau))
    f = 1. / tau

    J1,J2 = normcompliance_YT16(tau,Th)
    M1,M2 = compliance2modulus(J1,J2)

    arr_out =  np.zeros((n_tau,3))
    arr_out[:,0] = f
    arr_out[:,1] = M1
    arr_out[:,2] = M2

    header = '{0:^7s},{1:^7s},{2:^7s}\n{3:^7s},{4:^7s},{5:^7s}'.format('f','G_stor','G_loss','Hz','GPa','GPa')  # freq. is actually normalised
    file_out = os.path.join(f_out, f"YT16_Th_{Th:.2f}.txt")
    np.savetxt(file_out,arr_out,fmt=('%5.4e','%5.4e','%5.4e'),comments='',header=header,delimiter=",")

    data = visco.load.file(file_out)
    RefT=float(T_input)
    domain = 'freq'
    modul = 'G'
    df_master, units = visco.load.user_master(data, domain, RefT, modul)
    win = 1
    df_master = visco.master.smooth(df_master, win)
    df_dis = visco.prony.discretize(df_master,window='exact',nprony=n_prony)
    prony, df_GMaxw = visco.prony.fit(df_dis,df_master,opt=True)
    plot_fit(df_master,df_GMaxw,units,Th)
    file_pronyout = os.path.join(f_out,f"prony_coeff_YT16_Th_{Th:.2f}_N_{n_prony}.txt")
    prony['df_terms'].to_csv(file_pronyout,index=False)

    return prony['df_terms']

def track_coeff_Thdep():

    n_prony = 20
    n_Th = 100
    arr_Th = np.linspace(0.8,1.1,n_Th)
    alpha = np.zeros((3,n_Th,n_prony)) # (Th/tau/alpha) x (Th_i) x (Nprony_i)

    for i in range(n_Th):

        theta = arr_Th[i]
        file_pronyout = os.path.join(f_out,f"prony_coeff_YT16_Th_{theta:.2f}_N_{n_prony}.txt")

        if os.path.exists(file_pronyout):
            coeff = np.loadtxt(file_pronyout,delimiter=',',skiprows=1)
        else:
            coeff = fit_prony(theta,n_prony).to_numpy()

        alpha[0,i,:] = np.zeros(n_prony) + theta
        alpha[1,i,:] = coeff[:,0]
        alpha[2,i,:] = coeff[:,1]

    return alpha

def plot_coeff_Thdep():

    n_prony = 20
    n_Th = 10
    arr_Th = np.linspace(0.8,1.1,n_Th)
    alpha = track_coeff_Thdep()

    fig,ax=plt.subplots(4,5,figsize=(4*4,4*5))
    ax=ax.flatten()
    for i in range(n_prony):
        ax[i].plot(alpha[0,:,i],alpha[2,:,i],color='k',label=fr'$i=${i+1},$\tau_i=${np.log10(alpha[1,0,i]):.3f}')
        ax[i].legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(f_plot,"coeff_Thdep.jpg"),dpi=600)

plot_coeff_Thdep()