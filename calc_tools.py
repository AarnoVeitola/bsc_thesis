import numpy as np


# Calculates the flux surface average of A using the trapezoidal rule.
# It is assumed shat A = A(t, ky, z).
def fluxSurfaceAvg(A, J, z):
    return np.trapz(A * J.reshape((1,1,len(J))), z, axis=2) / np.trapz(J, z)


# Calculates the flux using the fluxSurfaceAvg function. M is the corresponding
# moment of the distribution fuction and phi the electrostatic potential. It is assumed that 
# phi = phi(t, kx, ky, z) and M = (t, kx, ky, z).
def calculateFlux(ky, z, phi, J, M, C):
    return fluxSurfaceAvg(np.sum(2 * np.real(1j * ky.reshape((1,1,len(ky),1)) * np.conjugate(phi) * M), axis=1) / C.reshape((1,len(z))), J, z)


# Calculates the flux surface average of the perpendicular wave number squared weighted by the electrostatic potential.
# Integrals are approximated using the trapezoidal rule. It is assumed that phi_amplitude = phi_amplitude(t, kx, ky, z),
# kperp_2 = kperp_2(kx, ky, z) and J = J(z).
def calculate_kperp_2_avg(kperp_2, phi_amplitude, J, kx, ky, z):
    nkx = len(kx)
    nky = len(ky)
    nz = len(z)
    numerator = np.sum(np.trapz(kperp_2.reshape((1,nkx,nky,nz)) * phi_amplitude * J.reshape((1,1,1,nz)), z, axis=3), axis=1)
    denominator = np.sum(np.trapz(phi_amplitude * J.reshape((1,1,1,nz)), z, axis=3), axis=1)
    return numerator / denominator


# Calculates the perpendicular wave number squared given kx, ky and z, and the components of the metric tensor.
def calculate_kperp_2(kx, ky, z, gxx, gxy, gyy):
    Kx, Ky, Z = np.meshgrid(kx, ky, z, indexing='ij')
    return np.power(Kx, 2) * gxx + 2 * Kx * Ky * gxy + np.power(Ky, 2) * gyy
