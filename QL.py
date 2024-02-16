from quasilinear import calculateQL
import numpy as np
from matplotlib import pyplot as plt


run_thesis_plots = True


def main():
    dir_path = '/scratch/rsalomaa/veitolaa'
    species = [
        'Hydrogen', 
        'Deuterium', 
        'Tritium', 
        'Hydrogen\n(high res.)'
        ]
    data_dirs = [
        'jet_hydrogen_ES', 
        'jet_deuterium_ES', 
        'jet_tritium_ES', 
        'jet_hydrogen_ES_hres'
        ]
    geom_name = 'tracer_efit'
    time_steps = [61, 91, 141, 16]
    eval_times = [60, 90, 140, 15]

    sim_data        = [None] * len(data_dirs)
    time            = [None] * len(data_dirs)
    QL              = [None] * len(data_dirs)
    L               = [None] * len(data_dirs)
    growth_rates    = [None] * len(data_dirs)
    frequencies     = [None] * len(data_dirs)
    potential       = [None] * len(data_dirs)

    for i in range(0, len(data_dirs)):
        (
            sim_data[i], time[i], QL[i], L[i], 
            growth_rates[i], frequencies[i], 
            potential[i]
            ) = calculateQL(dir_path, data_dirs[i], geom_name, time_steps[i])

    ########################
    ### PLOTS FOR THESIS ###
    ########################

    markers = ['s', 'o', '^', 'v']
    colors = ['tab:purple', 'tab:cyan', 'tab:pink', 'tab:brown']
    linestyles = ['solid', 'dashed', (0,(1,1))]

    nky = sim_data[0].parameters.nky0
    nky_i = nky - 12

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6,4))
    axs = [ax1, ax2, ax3, ax4]
    for i in range(0, len(data_dirs)):
        ax = axs[i]
        t = time[i]
        ax.axvline(t[eval_times[i]], color=colors[i], linestyle='dashed')
        Gamma_L = L[i][0]
        Q_L = L[i][1]      
        ax.plot(t, np.average(Gamma_L[:, 0], axis=1), color=colors[i])
        ax_twin = ax.twinx()
        ax_twin.plot(t, np.average(Q_L[:, 0], axis=1), color=colors[i])
        ax.set_yscale('symlog')
        ax_twin.set_yscale('symlog')
    fig.tight_layout()

    # Growth rates
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(6,4))
    ax1.axhline(0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    ax3.axhline(0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    ax1.axvline(1.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    ax3.axvline(1.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    for i in range(0, len(data_dirs) - 1):
        ky = sim_data[i].coordinates.ky
        gamma = growth_rates[i][0]
        gamma_err = growth_rates[i][1]
        omega = frequencies[i][0]
        omega_err = frequencies[i][1]
        ax1.plot(ky, gamma, color=colors[i], label=species[i], linestyle=linestyles[i])
        ax3.plot(ky, omega, color=colors[i], label=species[i], linestyle=linestyles[i])
        ax1.fill_between(ky, gamma+gamma_err, gamma-gamma_err, color=colors[i], alpha=0.3)
        ax3.fill_between(ky, omega+omega_err, omega-omega_err, color=colors[i], alpha=0.3)
    ax1.set_xlabel(r'$k_y \rho_i$')
    ax3.set_xlabel(r'$k_y \rho_i$')
    ax1.set_ylabel(r'$\gamma L_{{ref}} / c_{{ref}}$')
    ax3.set_ylabel(r'$\omega L_{{ref}} / c_{{ref}}$')
    ax3.legend(loc='lower left', fontsize='8')
    
    ky = sim_data[3].coordinates.ky
    gamma = growth_rates[3][0]
    gamma_err = growth_rates[3][1]
    omega = frequencies[3][0]
    omega_err = frequencies[3][1]
    ax2.axhline(0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    ax4.axhline(0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    ax2.axvline(1.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    ax4.axvline(1.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    ax2.plot(ky, gamma, label=species[3], color=colors[3])
    ax4.plot(ky, omega, label=species[3], color=colors[3])
    ax2.fill_between(ky, gamma+gamma_err, gamma-gamma_err, color=colors[3], alpha=0.3)
    ax4.fill_between(ky, omega+omega_err, omega-omega_err, color=colors[3], alpha=0.3)
    ax2.set_xlabel(r'$k_y \rho_i$')
    ax4.set_xlabel(r'$k_y \rho_i$')
    ax2.set_ylabel(r'$\gamma L_{{ref}} / c_{{ref}}$')
    ax4.set_ylabel(r'$\omega L_{{ref}} / c_{{ref}}$')
    ax4.legend(loc='lower left', fontsize='8')

    fig.tight_layout()

    plt.savefig('thesis_pictures/gamma.pdf', dpi='figure', format='pdf')
    plt.savefig('thesis_pictures/gamma.png')

    # QL flux comparison
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(8,4))
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel(r'$k_y \rho_i$')
    for i in range(0, len(data_dirs) - 1):
        if i == 0:
            axs = (ax4, ax5, ax6)
        else:
            axs = (ax1, ax2, ax3)
        t = eval_times[i]
        ky = sim_data[i].coordinates.ky
        nky = sim_data[i].parameters.nky0
        Gamma = QL[i][0]
        Q = QL[i][1]
        axs[0].plot(ky[:nky_i], Gamma[t, 0, :nky_i] / np.sum(Q[t, 1]), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[1].plot(ky[:nky_i], Q[t, 0, :nky_i] / np.sum(Q[t, 1]), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[2].plot(ky[:nky_i], Q[t, 1, :nky_i] / np.sum(Q[t, 1]), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
    
    ax1.set_ylabel(r'$\Gamma_e(k_y) / Q_i$')
    ax2.set_ylabel(r'$Q_e(k_y) / Q_i$')
    ax3.set_ylabel(r'$Q_i(k_y) / Q_i$')
    ax4.set_ylabel(r'$\Gamma_e(k_y) / Q_i$')
    ax5.set_ylabel(r'$Q_e(k_y) / Q_i$')
    ax6.set_ylabel(r'$Q_i(k_y) / Q_i$')
    ax1.legend(loc='lower left', fontsize='8')
    ax4.legend(loc='lower left', fontsize='8')

    fig.tight_layout()

    plt.savefig('thesis_pictures/QL_fluxes.pdf', dpi='figure', format='pdf')
    plt.savefig('thesis_pictures/QL_fluxes.png')

    # QL flux comparison presentation
    fig, (axs1, axs2, axs3) = plt.subplots(3, 3, figsize=(8,6))
    for axs in [axs1, axs2, axs3]:
        for ax in axs:
            ax.axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
            ax.set_xlabel(r'$k_y \rho_i$')
    for i in range(0, len(data_dirs) - 1):
        if i == 0:
            axs = axs1
        elif i == 1:
            axs = axs2
        else:
            axs = axs3
        t = eval_times[i]
        ky = sim_data[i].coordinates.ky
        nky = sim_data[i].parameters.nky0
        Gamma = QL[i][0]
        Q = QL[i][1]
        axs[0].plot(ky[:nky_i], Gamma[t, 0, :nky_i] / np.sum(Q[t, 1]), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[1].plot(ky[:nky_i], Q[t, 0, :nky_i] / np.sum(Q[t, 1]), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[2].plot(ky[:nky_i], Q[t, 1, :nky_i] / np.sum(Q[t, 1]), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
    
        axs[0].set_ylabel(r'$\Gamma_e(k_y) / Q_i$')
        axs[1].set_ylabel(r'$Q_e(k_y) / Q_i$')
        axs[2].set_ylabel(r'$Q_i(k_y) / Q_i$')
        axs[0].legend(loc='lower left', fontsize='8')

    fig.tight_layout()

    plt.savefig('thesis_pictures/QL_fluxes_pres.pdf', dpi='figure', format='pdf')
    plt.savefig('thesis_pictures/QL_fluxes_pres.png')

    # QL weights
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,2.5))
    ax1.axvline(1.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    ax2.axvline(1.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
    for i in range(0, len(data_dirs)):
        ky = sim_data[i].coordinates.ky
        wQL = QL[i][2]
        t = eval_times[i]
        if i == 3:
            ax2.plot(ky, wQL[t], label=species[i], color=colors[i], 
            marker=markers[i], markerfacecolor='None', markersize=5)
        else:
            ax1.plot(ky, wQL[t], label=species[i], color=colors[i],
            marker=markers[i], markerfacecolor='None', markersize=5)
    ax1.set_xlabel(r'$k_y \rho_i$')
    ax2.set_xlabel(r'$k_y \rho_i$')
    ax1.set_ylabel(r'$w^{QL}(k_y)$')
    ax2.set_ylabel(r'$w^{QL}(k_y)$')
    ax1.legend(loc='upper left', fontsize='8')
    ax2.legend(loc='upper left', fontsize='8')

    fig.tight_layout()

    plt.savefig('thesis_pictures/QL_weights.pdf', dpi='figure', format='pdf')
    plt.savefig('thesis_pictures/QL_weights.png')

    plt.show()


main()
