import numpy as np
from scipy.io import FortranFile


# Reading data from mom_.dat files. Returns the time 
# vector, and complex arrays for n, Tpar and Tperp.
def readMom(file_path, time_steps, nkx, nky, nz):
    f = FortranFile(file_path, 'r')
    time        = np.empty((time_steps,), dtype=float)
    dens        = np.empty((time_steps,nkx,nky,nz), dtype=complex)
    temp_par    = np.empty((time_steps,nkx,nky,nz), dtype=complex)
    temp_perp   = np.empty((time_steps,nkx,nky,nz), dtype=complex)

    t = 0
    while t < time_steps:
        time[t]         = f.read_reals(float)
        dens[t]         = f.read_reals(complex).reshape((nkx,nky,nz), order='F')
        temp_par[t]     = f.read_reals(complex).reshape((nkx,nky,nz), order='F')
        temp_perp[t]    = f.read_reals(complex).reshape((nkx,nky,nz), order='F')
        # Skip entries
        f.read_reals() # q1 + 1.5 p_0 u1par
        f.read_reals() # q1perp + p_0 u1par
        f.read_reals() # u1par
        t += 1
    f.close()
    return time, dens, temp_par, temp_perp


# Reading data from field.dat files. Returns the complex potential phi = phi(t, kx, ky, z) 
# and the time vector that phi is evaluated for. n_fields defines the number of fields in 
# the field.dat file.
def readField(file_path, time_steps, nkx, nky, nz, n_fields):
    f = FortranFile(file_path, 'r')
    time =  np.empty((time_steps,), dtype=float)
    field = np.empty((time_steps,nkx,nky,nz), dtype=complex)

    t = 0
    while t < time_steps:
        time[t]     = f.read_reals(float)
        field[t]    = f.read_reals(complex).reshape((nkx,nky,nz), order='F')
        for i in range(1, n_fields):
            f.read_reals() # Skip redundant fields
        t += 1
        
    f.close()
    return time, field


# Reading data from .dat files
def readNrg(file_path, time_steps, ns):
    time =  np.empty((time_steps,), dtype=float)
    Gamma_es = np.empty((time_steps,ns), dtype=float)
    Q_es     = np.empty((time_steps,ns), dtype=float)
    with open(file_path, 'r') as f:
        for t in range(0, time_steps):
            time[t] = float(f.readline().rstrip())
            for s_i in range(0, ns):
                entry = np.fromstring(f.readline(), dtype=float, sep=' ')
                Gamma_es[t, s_i] = entry[4]
                Q_es[t, s_i]     = entry[6]
    return time, Gamma_es, Q_es
    

# Reading the <geometry name>.dat file
def readGeometry(file_path, nz):
    with open(file_path, 'r') as f:
        line = f.readline()
        # Read skip over parameters
        while line.strip() != '/':
            line = f.readline()
        
        entry = np.fromstring(f.readline(), dtype=float, sep=' ')
        data = np.ndarray(shape=(nz,len(entry)), dtype=float)
        data[0] = entry
        for i in range(1, nz):
            data[i] = np.fromstring(f.readline(), dtype=float, sep=' ')
        return data


# Reading the frequency_act.dat file
def readFrequency(file_path, nky):
    data = np.ndarray(shape=(nky,5), dtype=float)
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            if line[0] != '#':
                data[i] = np.fromstring(line, dtype=float, sep=' ')
                i += 1
        return data


# Loading parameters from parameters.dat file. Returns a dictionary containing 
# the parameters with the names found in param_names.
def readParameters(file_path, param_names):
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            key_value = line.split('=')
            if len(key_value) == 2 and key_value[0].strip() in param_names:
                params[key_value[0].strip()] = float(key_value[1].strip())
    return params


# Loading species from parameter.dat file. Returns the species as a list of type Specie.
# s_i in the index of the specie to be loaded.
def readSpecies(file_path, s_i):
    i = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '&species':
                name =      f.readline().split('=')[-1].strip().replace('\'', '')
                omn =       float(f.readline().split('=')[-1].strip())
                omt =       float(f.readline().split('=')[-1].strip())
                f.readline()
                mass =      float(f.readline().split('=')[-1].strip())
                temp =      float(f.readline().split('=')[-1].strip())
                dens =      float(f.readline().split('=')[-1].strip())
                charge =    float(f.readline().split('=')[-1].strip())
                if i == s_i:
                    return {
                        'name': name, 'omn': omn, 
                        'omt': omt, 'mass': mass, 
                        'temp': temp, 'dens': dens, 
                        'charge': charge
                        }
                i += 1
    