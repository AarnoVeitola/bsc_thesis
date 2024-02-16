import numpy as np
from scipy.constants import pi
from data_tools import readParameters, readSpecies, readGeometry


# An object that contains the simulation parameters
class Parameters:
    def __init__(self, param_path, geom_path):
        # List of parameters to be loaded
        param_names = [
            'Bref', 'Tref', 'nref', 'Lref', 
            'mref', 'lx', 'ly', 'kymin', 
            'nx0', 'nky0', 'nz0', 'n_spec', 
            'n_fields', 'step_time', 'istep_field', 
            'istep_mom', 'istep_nrg', 'rhostar',
            'major_R', 'minor_r', 'number of computed time steps'
            ]  
        # Load parameters
        params = readParameters(param_path, param_names)
        # Set parameters
        self.Bref       = params['Bref']        # Reference magnetic field in T
        self.Tref       = params['Tref']        # Reference temperature in keV
        self.nref       = params['nref']        # Reference density in 10^19 m^-3
        self.Lref       = params['Lref']        # Reference length in m
        self.mref       = params['mref']        # Reference mass in units of proton mass
        self.lx         = params['lx']          # Extension of the simulation box in the x directions
        self.ly         = params['ly']          # Extension of the simulation box in the y directions
        self.kymin      = params['kymin']       # The minimum value for ky
        self.step_time  = params['step_time']   # Simulation time step in s
        self.rhostar    = params['rhostar']     # rhostar = rhoref / a
        self.major_R    = params['major_R']     # Major radius R
        self.minor_r    = params['minor_r']     # Minor radius a
        self.nx0                = int(params['nx0'])            # Number of grid points in x dir
        self.nky0               = int(params['nky0'])           # Number of Fourier modes in the y direction
        self.nz0                = int(params['nz0'])            # Number of grid points in z dir
        self.n_spec             = int(params['n_spec'])         # Number of species
        self.n_fields           = int(params['n_fields'])       # Number of fields
        self.istep_field        = int(params['istep_field'])    # Number of steps between entries in field.dat
        self.istep_mom          = int(params['istep_mom'])      # Number of steps between entries in mom.dat
        self.istep_nrg          = int(params['istep_nrg'])      # Number of steps between entries in nrg.dat
        self.no_comp_time_steps = int(params['number of computed time steps']) # Number of time steps

        # Geometry parameters to be loaded
        geom_param_names = ['Cxy']
        # Load geometry parameters
        geom_params = readParameters(geom_path, geom_param_names)
        # Set geometry parameters
        self.Cxy = geom_params['Cxy']
    
    def __str__(self):
        return 'An object containing simulation parameters'


# An object that contains the simulation geometry
class Geometry:
    def __init__(self, geom_path, nz0):
        # Load geometry data
        geom_data = readGeometry(geom_path, nz0)
        # Set geometry data
        self.gxx = geom_data[:,0]
        self.gxy = geom_data[:,1]
        self.gyy = geom_data[:,3]
        self.B0  = geom_data[:,6]
        self.J   = geom_data[:,10]

    def __str__(self):
        return 'An object containing a simulation geometry'


# An object that contains the simulation coordinates
class Coordinates:
    def __init__(self, nx0, nky0, nz0, lx, kymin):
        self.kx = np.arange(-nx0/2+1, nx0/2+1, 1) * 2 * pi / lx
        self.ky = np.arange(1, nky0+1, 1) * kymin
        self.z  = np.linspace(-pi, pi, nz0)

    def __str__(self):
        return 'An object cotaining simulation coordinates'


# An object containing the data of a certain species
class Specie:
    def __init__(self, param_path, s_i):
        # Load specie data
        spec = readSpecies(param_path, s_i)
        # Set data
        self.name   = spec['name']
        self.omn    = spec['omn']
        self.omt    = spec['omt']
        self.mass   = spec['mass']
        self.temp   = spec['temp']
        self.dens   = spec['dens']
        self.charge = spec['charge']

    def __str__(self):
        return f'''
        name = {self.name}\n
        omega_n = {self.omn}\n
        omega_t = {self.omt}\n
        mass = {self.mass}\n
        temperature = {self.temp}\n
        density = {self.dens}\n
        charge = {self.charge}
        '''


# An object containing the data of a certain simulation
class General:
    def __init__(self, param_path, geom_path):
        # Parameters and geometry
        self.parameters = Parameters(param_path, geom_path)
        self.geometry = Geometry(geom_path, self.parameters.nz0)
        # Coordinates
        self.coordinates = Coordinates(
            self.parameters.nx0, 
            self.parameters.nky0, 
            self.parameters.nz0, 
            self.parameters.lx, 
            self.parameters.kymin
            )
        # Species
        self.species = []
        for s_i in range(0, self.parameters.n_spec):
            self.species.append(Specie(param_path, s_i))
        
    
    def __str__(self):
        return 'An object containing the simulation output'
        