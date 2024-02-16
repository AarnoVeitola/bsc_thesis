import numpy as np
from data_tools import readField, readMom, readNrg
from calc_tools import calculateFlux
from sim import General


def calculateNL(dir_path: str, data_dir: str, geom_name: str, max_time_steps=-1, file_version='.dat'):
    # Set data path
    data_path = f'{dir_path}/{data_dir}'
    print(f'Reading data from {data_path}')
    
    # Define file paths
    param_path  = f'{data_path}/parameters{file_version}'
    field_path  = f'{data_path}/field{file_version}'
    geom_path   = f'{data_path}/{geom_name}{file_version}'
    nrg_path    = f'{data_path}/nrg{file_version}'

    # Load simulation data
    general = General(param_path, geom_path)
    print(f'{param_path} processed')
    print(f'{geom_path} processed')

    # Parameters
    nx = general.parameters.nx0
    nky = general.parameters.nky0
    nz = general.parameters.nz0
    ns = general.parameters.n_spec
    n_fields = general.parameters.n_fields
    gxx = general.geometry.gxx
    gxy = general.geometry.gxy
    gyy = general.geometry.gyy
    B0 = general.geometry.B0
    J = general.geometry.J
    istep_field = general.parameters.istep_field
    istep_mom = general.parameters.istep_mom
    istep_nrg = general.parameters.istep_nrg
    no_time_steps = general.parameters.no_comp_time_steps

    # Coordinates
    kx = general.coordinates.kx
    ky = general.coordinates.ky
    z  = general.coordinates.z

    # Time
    mom_field_ratio = int(istep_mom / istep_field) # Calculate ratio for mom and field file entries
    field_steps = int(no_time_steps / istep_field)
    mom_steps = int(no_time_steps / istep_mom)
    nrg_steps = int(no_time_steps / istep_nrg)
    if mom_steps != mom_field_ratio * field_steps :
        field_steps = int(mom_steps * mom_field_ratio)

    if max_time_steps != -1:
        if mom_field_ratio >= 1.0:
            # If field has more entries
            field_steps = int(max_time_steps * mom_field_ratio)
            mom_steps = max_time_steps
        else:
            # If mom has more entries
            field_steps = max_time_steps
            mom_steps = int(max_time_steps / mom_field_ratio)
        # Set time step resolution
    time_steps_res = np.min([field_steps, mom_steps])

    # Process field data
    time_field, phi = readField(field_path, field_steps, nx, nky, nz, n_fields)
    # Slicing (assuming istep_mom >= istep_field)
    if mom_field_ratio >= 1:
        time_field = time_field[0::mom_field_ratio]
        phi = phi[0::mom_field_ratio]
    print(f'{field_path} processed')

    # Read nrg data
    time_nrg, Gamma_avg, Q_avg = readNrg(nrg_path, nrg_steps, ns)
    print(f'{nrg_path} processed')

    # Calculations
    C = B0 / np.sqrt(gxx * gyy + np.square(gxy))
    
    # Initialize non-linear fluxes
    Gamma_NL   = np.zeros(shape=(time_steps_res,ns,nky), dtype=float)
    Q_NL       = np.zeros(shape=(time_steps_res,ns,nky), dtype=float)

    # Calculating the fluxes
    for s_i in range(0, ns):    # Iterate over all species
        specie = general.species[s_i]
        n0 = specie.dens
        T0 = specie.temp
        mom_path = f'{data_path}/mom_{specie.name}{file_version}'
        # Load mom data
        time_mom, n, Tpar, Tperp = readMom(mom_path, mom_steps, nx, nky, nz)
        # Slicing
        if mom_field_ratio < 1.0:
            time_mom = time_mom[0::int(1/mom_field_ratio)]
            n = n[0::int(1/mom_field_ratio)]
            Tpar = Tpar[0::int(1/mom_field_ratio)]
            Tperp = Tperp[0::int(1/mom_field_ratio)]
        print(f'{mom_path} processed')
        
        # Calculate non-linear fluxes
        Gamma_NL[:, s_i]   = calculateFlux(ky, z, phi, J, n, C) 
        Q_NL[:, s_i]       = calculateFlux(ky, z, phi, J, 0.5*Tpar*n0+Tperp*n0+1.5*n*T0, C)

    # Set return time vector
    if len(time_field) < len(time_mom):
        time = time_field
    else:
        time = time_mom

    return general, (time, Gamma_NL, Q_NL), (time_nrg, Gamma_avg, Q_avg)
