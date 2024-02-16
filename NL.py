import numpy as np
from matplotlib import pyplot as plt
from nonlinear import calculateNL
from scipy.optimize import curve_fit


def main():
    dir_path = '/scratch/rsalomaa/veitolaa'
    species = [
        'Hydrogen', 
        'Deuterium', 
        'Tritium'
        ]
    data_dirs = [
        'jet_hydrogen_NL', 
        'jet_deuterium_NL', 
        'jet_tritium_NL'
        ]
    geom_name = 'tracer_efit'

    file_versions = [
        ['_001', '_002', '.dat'], 
        ['_001', '_002', '.dat'], 
        ['_001', '_002', '_003', '.dat']
        ]

    sim_data    = [None] * len(data_dirs)
    NL_times    = [None] * len(data_dirs)
    nrg_times   = [None] * len(data_dirs)
    Gamma       = [None] * len(data_dirs)
    Q           = [None] * len(data_dirs)
    Gamma_avg   = [None] * len(data_dirs)
    Q_avg       = [None] * len(data_dirs)
    

    for i in range(0, len(data_dirs)):
        for j in range(0, len(file_versions[i])):
            general, NL, NRG = calculateNL(
                dir_path, data_dirs[i], geom_name,
                file_version=file_versions[i][j]
                )
            if j == 0:
                sim_data[i] = general
                NL_times[i] = NL[0]
                Gamma[i] = NL[1]
                Q[i] = NL[2]
                nrg_times[i] = NRG[0]
                Gamma_avg[i] = NRG[1]
                Q_avg[i] = NRG[2]
            else:
                NL_times[i] = np.append(NL_times[i], NL[0])
                Gamma[i] = np.append(Gamma[i], NL[1], axis=0)
                Q[i] = np.append(Q[i], NL[2], axis=0)
                nrg_times[i] = np.append(nrg_times[i], NRG[0])
                Gamma_avg[i] = np.append(Gamma_avg[i], NRG[1], axis=0)
                Q_avg[i] = np.append(Q_avg[i], NRG[2], axis=0)


    markers = ['s', 'o', '^', 'v']
    colors = ['tab:purple', 'tab:cyan', 'tab:pink', 'tab:brown']
    sat_times = [None] * len(data_dirs)
    sat_times_nrg = [None] * len(data_dirs)
    for i in range(0, len(data_dirs)):
        sat_times[i] = np.argmin(np.abs(NL_times[i] - 75.0))
        sat_times_nrg[i] = np.argmin(np.abs(nrg_times[i] - 75.0))
    nky_i = sim_data[0].parameters.nky0 - 12
    
    D_e = np.array([1.06, 0.69])
    V_e = np.array([0.46, 0.14])
    Gamma_exp = (sim_data[0].species[0].dens / sim_data[0].parameters.major_R) * (D_e * sim_data[0].parameters.major_R * sim_data[0].species[0].omn + sim_data[0].parameters.major_R * V_e)
    print(Gamma_exp)

    Q_e_exp = [39, 28]
    Q_i_exp = [60, 52]
    Gamma_exp = [12, 8]

    # Calculating total fluxes
    ns = sim_data[0].parameters.n_spec
    total_fluxes = np.ndarray(shape=(len(data_dirs),ns,2), dtype=float)
    scaling_factors = np.ndarray(shape=(ns,2), dtype=float)
    for i in range(0, len(data_dirs)):
        time = nrg_times[i]
        print(time[sat_times_nrg[i]])
        for j in range(0, ns):
            total_fluxes[i, j, 0] = np.trapz(Gamma_avg[i][sat_times_nrg[i]:, j], 
            time[sat_times_nrg[i]:]) / (time[-1] - time[sat_times_nrg[i]])
            total_fluxes[i, j, 1] = np.trapz(Q_avg[i][sat_times_nrg[i]:, j], 
            time[sat_times_nrg[i]:]) / (time[-1] - time[sat_times_nrg[i]])

    # Calculating mass number scaling
    A = [1,2,3]
    labels = [[r'$\Gamma_{e, tot}/\Gamma_{gb}$', r'$\Gamma_{i, tot}/\Gamma_{gb}$'], [r'$Q_{e, tot}/Q_{gb}$', r'$Q_{i, tot}/Q_{gb}$']]
    fig, ax = plt.subplots()
    for i in range(0, ns):
        A = [1, 2, 3]
        data = total_fluxes[:, i]
        def func(A, a, b, c):
            return a * np.power(A, b) + c
        for j in range(0, 2):
            popt, pcov = curve_fit(func, A, data[:, j])
            a, b, c = popt
            scaling_factors[i, j] = b
            if i != 1 or j != 0:
                ax.scatter(A, total_fluxes[:, i, j], label=labels[j][i], marker='+')
                ax.plot(A, func(A, a, b, c), label=r'${:.2f} \cdot A^{{:.2f}} + {:.2f}$'.format(a, b, c))
    ax.legend(loc='upper right')
    print(scaling_factors)
    print(total_fluxes)
    
    

    # Flux saturation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,5))
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel(r'$t/(L_{{ref}}/c_s)$')
    for i in range(0, len(data_dirs)):
        if i in [1, 2]:
            gB_factor = np.square(sim_data[i].parameters.major_R)
            ax1.axhline(Gamma_exp[i-1] / gB_factor, color=colors[i],
            label=f'{species[i]} exp.', linestyle='dashed')
            ax2.axhline(Gamma_exp[i-1] / gB_factor, color=colors[i], 
            label=f'{species[i]} exp.', linestyle='dashed')
            ax3.axhline(Q_e_exp[i-1] / gB_factor, color=colors[i],
            label=f'{species[i]} exp.', linestyle='dashed')
            ax4.axhline(Q_i_exp[i-1] / gB_factor, color=colors[i], 
            label=f'{species[i]} exp.', linestyle='dashed')
        ax1.plot(nrg_times[i], Gamma_avg[i][:, 0], color=colors[i], label=species[i])
        ax2.plot(nrg_times[i], Gamma_avg[i][:, 1], color=colors[i], label=species[i])
        ax3.plot(nrg_times[i], Q_avg[i][:, 0], color=colors[i])
        ax4.plot(nrg_times[i], Q_avg[i][:, 1], color=colors[i])
    ax1.set_ylabel(r'$\Gamma_e/\Gamma_{{gb}}$')
    ax2.set_ylabel(r'$\Gamma_i/\Gamma_{{gb}}$')
    ax3.set_ylabel(r'$Q_e/Q_{{gb}}$')
    ax4.set_ylabel(r'$Q_i/Q_{{gb}}$')
    ax1.legend(loc='upper left', fontsize=8)

    fig.tight_layout()

    plt.savefig('thesis_pictures/NL_fluxes.pdf', dpi='figure', format='pdf')
    plt.savefig('thesis_pictures/NL_fluxes.png')

    # Flux saturation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,5))
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel(r'$t/(L_{{ref}}/c_s)$')
    for i in range(0, len(data_dirs)):
        ax1.plot(NL_times[i], np.sum(Gamma[i][:, 0], axis=1), color=colors[i], label=species[i])
        ax2.plot(NL_times[i], np.sum(Q[i][:, 0], axis=1), color=colors[i])
        ax3.plot(NL_times[i], np.sum(Gamma[i][:, 1], axis=1), color=colors[i], label=species[i])
        ax4.plot(NL_times[i], np.sum(Q[i][:, 1], axis=1), color=colors[i])
    ax1.set_ylabel(r'$\Gamma_e/\Gamma_{{gb}}$')
    ax2.set_ylabel(r'$Q_e/Q_{{gb}}$')
    ax3.set_ylabel(r'$\Gamma_i/\Gamma_{{gb}}$')
    ax4.set_ylabel(r'$Q_i/Q_{{gb}}$')
    ax1.legend(loc='upper left', fontsize=8)

    fig.tight_layout()

    # Flux spectra
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(8,4))
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axvline(1.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
        ax.axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel(r'$k_y \rho_i$')
    for i in range(0, len(data_dirs)):
        if i == 0:
            axs = [ax4, ax5, ax6]
        else:
            axs = [ax1, ax2, ax3]
        ky = sim_data[i].coordinates.ky
        axs[0].plot(ky, np.average(Gamma[i][sat_times[i]:, 0], axis=0), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[1].plot(ky, np.average(Q[i][sat_times[i]:, 0], axis=0), color=colors[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[2].plot(ky, np.average(Q[i][sat_times[i]:, 1], axis=0), color=colors[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
    ax1.set_ylabel(r'$\Gamma_e/\Gamma_{{gb}}$')
    ax2.set_ylabel(r'$Q_e/Q_{{gb}}$')
    ax3.set_ylabel(r'$Q_i/Q_{{gb}}$')
    ax4.set_ylabel(r'$\Gamma_e/\Gamma_{{gb}}$')
    ax5.set_ylabel(r'$Q_e/Q_{{gb}}$')
    ax6.set_ylabel(r'$Q_i/Q_{{gb}}$')
    ax1.legend(loc='upper right', fontsize=8)
    ax4.legend(loc='upper right', fontsize=8)

    fig.tight_layout()

    plt.savefig('thesis_pictures/NL_flux_spectra.pdf', dpi='figure', format='pdf')
    plt.savefig('thesis_pictures/NL_flux_spectra.png')

    # Flux spectra 2
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(8,4))
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel(r'$k_y \rho_i$')
    for i in range(0, len(data_dirs)):
        if i == 0:
            axs = [ax4, ax5, ax6]
        else:
            axs = [ax1, ax2, ax3]
        ky = sim_data[i].coordinates.ky
        axs[0].plot(ky[:nky_i], np.average(Gamma[i][sat_times[i]:, 0, :nky_i], axis=0), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[1].plot(ky[:nky_i], np.average(Q[i][sat_times[i]:, 0, :nky_i], axis=0), color=colors[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[2].plot(ky[:nky_i], np.average(Q[i][sat_times[i]:, 1, :nky_i], axis=0), color=colors[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
    ax1.set_ylabel(r'$\Gamma_e/\Gamma_{{gb}}$')
    ax2.set_ylabel(r'$Q_e/Q_{{gb}}$')
    ax3.set_ylabel(r'$Q_i/Q_{{gb}}$')
    ax4.set_ylabel(r'$\Gamma_e/\Gamma_{{gb}}$')
    ax5.set_ylabel(r'$Q_e/Q_{{gb}}$')
    ax6.set_ylabel(r'$Q_i/Q_{{gb}}$')
    ax1.legend(loc='upper right', fontsize=8)
    ax4.legend(loc='upper right', fontsize=8)

    fig.tight_layout()

    plt.savefig('thesis_pictures/NL_flux_spectra_2.pdf', dpi='figure', format='pdf')
    plt.savefig('thesis_pictures/NL_flux_spectra_2.png')

    # Flux spectra 2 presentation
    fig, (axs1, axs2, axs3) = plt.subplots(3, 3, figsize=(8,6))
    for axs in [axs1, axs2, axs3]:
        for i in range(0, 3):
            axs[i].axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
            axs[i].set_xlabel(r'$k_y \rho_i$')
    for i in range(0, len(data_dirs)):
        if i == 0:
            axs = axs1
        elif i == 1:
            axs = axs2
        else:
            axs = axs3
        ky = sim_data[i].coordinates.ky
        axs[0].plot(ky[:nky_i], np.average(Gamma[i][sat_times[i]:, 0, :nky_i], axis=0), color=colors[i], label=species[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[1].plot(ky[:nky_i], np.average(Q[i][sat_times[i]:, 0, :nky_i], axis=0), color=colors[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[2].plot(ky[:nky_i], np.average(Q[i][sat_times[i]:, 1, :nky_i], axis=0), color=colors[i], 
        marker=markers[i], markerfacecolor='None', markersize=5)
        axs[0].set_ylabel(r'$\Gamma_e/\Gamma_{{gb}}$')
        axs[1].set_ylabel(r'$Q_e/Q_{{gb}}$')
        axs[2].set_ylabel(r'$Q_i/Q_{{gb}}$')
        axs[0].legend(loc='upper right', fontsize=8)

    fig.tight_layout()

    plt.savefig('thesis_pictures/NL_flux_spectra_2_pres.pdf', dpi='figure', format='pdf')
    plt.savefig('thesis_pictures/NL_flux_spectra_2_pres.png')

    plt.show()


main()
