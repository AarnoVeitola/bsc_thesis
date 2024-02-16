import numpy as np
from matplotlib import pyplot as plt
from nonlinear import calculateNL
from scipy.optimize import curve_fit


def main():
    dir_path = '/scratch/rsalomaa/veitolaa'
    species = [
        'Deuterium'
        ]
    data_dirs = [
        'jet_deuterium_NL'
        ]
    geom_name = 'tracer_efit'

    file_versions = [['_001']]

    sim_data    = [None] * len(data_dirs)
    NL_times    = [None] * len(data_dirs)
    nrg_times   = [None] * len(data_dirs)
    Gamma       = [None] * len(data_dirs)
    Q           = [None] * len(data_dirs)
    Gamma_avg   = [None] * len(data_dirs)
    Q_avg       = [None] * len(data_dirs)
    

    for i in range(0, len(data_dirs)):
        for j in range(0, len(file_versions[i])):
            general, NL, NRG = calculateNL(dir_path, data_dirs[i], geom_name, file_version=file_versions[i][j])
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
    sat_times = [97, 135, 163]
    

    # Flux saturation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,5))
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel(r'$t/(L_{{ref}}/c_s)$')
    for i in range(0, len(data_dirs)):
        # for ax in [ax1, ax2, ax3, ax4]:
        #     ax.axvline(time[i][sat_times[i]], color=colors[i], linestyle='dashed', linewidth=0.5)
        ax1.plot(nrg_times[i], Gamma_avg[i][:, 0], color=colors[i], label=species[i])
        ax2.plot(nrg_times[i], Q_avg[i][:, 0], color=colors[i])
        ax3.plot(nrg_times[i], Gamma_avg[i][:, 1], color=colors[i], label=species[i])
        ax4.plot(nrg_times[i], Q_avg[i][:, 1], color=colors[i])
    ax1.set_ylabel(r'$\Gamma_e/\Gamma_{{gb}}$')
    ax2.set_ylabel(r'$Q_e/Q_{{gb}}$')
    ax3.set_ylabel(r'$\Gamma_i/\Gamma_{{gb}}$')
    ax4.set_ylabel(r'$Q_i/Q_{{gb}}$')
    ax1.legend(loc='upper left', fontsize=8)

    fig.tight_layout()

    plt.savefig('thesis_pictures/NL_fluxes.pdf', dpi='figure', format='pdf')

    # Flux saturation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7,5))
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axhline(0.0, color='tab:grey', linestyle='dashed', linewidth=0.5)
        ax.set_xlabel(r'$t/(L_{{ref}}/c_s)$')
    for i in range(0, len(data_dirs)):
        # for ax in [ax1, ax2, ax3, ax4]:
        #     ax.axvline(time[i][sat_times[i]], color=colors[i], linestyle='dashed', linewidth=0.5)
        ax1.plot(NL_times[i], np.sum(Gamma[i][:, 0], axis=1), color=colors[i], label=species[i])
        ax2.plot(NL_times[i], np.sum(Q[i][:, 0], axis=1), color=colors[i])
        ax3.plot(NL_times[i], np.sum(Gamma[i][:, 1], axis=1), color=colors[i], label=species[i])
        ax4.plot(NL_times[i], np.sum(Q[i][:, 1], axis=1), color=colors[i])
    ax1.set_ylabel(r'$\Gamma_e$')
    ax2.set_ylabel(r'$Q_e$')
    ax3.set_ylabel(r'$\Gamma_i$')
    ax4.set_ylabel(r'$Q_i$')
    ax1.legend(loc='upper left', fontsize=8)

    fig.tight_layout()

    plt.show()

main()
