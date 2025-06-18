"""
Mathematical utilities for exciton-phonon calculations.

Contains special functions, distribution functions, and numerical utilities.
"""

import numpy as np
from scipy import special
from ..core.constants import PhysicalConstants

# Try to import numba, fall back to regular functions if not available
try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True)
def fermi_dirac(energy, mu, T):
    """
    Fermi-Dirac distribution function.
    
    f(E) = 1 / (exp((E-μ)/k_B T) + 1)
    
    Parameters:
    -----------
    energy : float or array
        Energy values
    mu : float
        Chemical potential
    T : float
        Temperature (K)
        
    Returns:
    --------
    float or array
        Fermi-Dirac distribution values
    """
    if T == 0:
        return np.where(energy <= mu, 1.0, 0.0)
    else:
        beta = 1.0 / (PhysicalConstants.k_B * T)
        x = beta * (energy - mu)
        # Avoid overflow for large x
        return np.where(x > 50, 0.0, 1.0 / (np.exp(x) + 1.0))


@jit(nopython=True)
def bose_einstein(energy, T):
    """
    Bose-Einstein distribution function.
    
    n(E) = 1 / (exp(E/k_B T) - 1)
    
    Parameters:
    -----------
    energy : float or array
        Energy values
    T : float
        Temperature (K)
        
    Returns:
    --------
    float or array
        Bose-Einstein distribution values
    """
    if T == 0:
        return np.zeros_like(energy)
    else:
        beta = 1.0 / (PhysicalConstants.k_B * T)
        x = beta * energy
        # Avoid overflow and division by zero
        return np.where(x > 50, 0.0, 
                       np.where(x < 1e-12, 1.0/x - 0.5,
                               1.0 / (np.exp(x) - 1.0)))


@jit(nopython=True)
def lorentzian(x, x0, gamma):
    """
    Lorentzian function.
    
    L(x) = (γ/π) / ((x-x₀)² + γ²)
    
    Parameters:
    -----------
    x : float or array
        Variable
    x0 : float
        Center position
    gamma : float
        Half-width at half-maximum
        
    Returns:
    --------
    float or array
        Lorentzian values
    """
    return (gamma / np.pi) / ((x - x0)**2 + gamma**2)


@jit(nopython=True)
def gaussian(x, x0, sigma):
    """
    Gaussian function.
    
    G(x) = (1/σ√(2π)) exp(-(x-x₀)²/(2σ²))
    
    Parameters:
    -----------
    x : float or array
        Variable
    x0 : float
        Center position
    sigma : float
        Standard deviation
        
    Returns:
    --------
    float or array
        Gaussian values
    """
    norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
    return norm * np.exp(-0.5 * ((x - x0) / sigma)**2)


def delta_function(x, x0, broadening=1e-3, shape='lorentzian'):
    """
    Approximate delta function using broadening.
    
    Parameters:
    -----------
    x : array
        Variable array
    x0 : float
        Delta function position
    broadening : float
        Broadening parameter
    shape : str
        Broadening shape ('lorentzian' or 'gaussian')
        
    Returns:
    --------
    array
        Approximate delta function
    """
    if shape == 'lorentzian':
        return lorentzian(x, x0, broadening)
    elif shape == 'gaussian':
        return gaussian(x, x0, broadening)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def integrate_1d(func, x_array):
    """
    Simple 1D integration using trapezoidal rule.
    
    Parameters:
    -----------
    func : array
        Function values
    x_array : array
        Coordinate array
        
    Returns:
    --------
    float
        Integral value
    """
    return np.trapz(func, x_array)


def integrate_2d(func, x_array, y_array):
    """
    2D integration using trapezoidal rule.
    
    Parameters:
    -----------
    func : 2D array
        Function values on 2D grid
    x_array : array
        x-coordinates
    y_array : array
        y-coordinates
        
    Returns:
    --------
    float
        Integral value
    """
    # Integrate along y first, then x
    integral_y = np.trapz(func, y_array, axis=1)
    integral_xy = np.trapz(integral_y, x_array)
    return integral_xy


@jit(nopython=True)
def complex_to_polar(z):
    """
    Convert complex number to polar form.
    
    Parameters:
    -----------
    z : complex
        Complex number
        
    Returns:
    --------
    tuple
        (magnitude, phase)
    """
    magnitude = np.abs(z)
    phase = np.angle(z)
    return magnitude, phase


@jit(nopython=True)
def polar_to_complex(magnitude, phase):
    """
    Convert polar form to complex number.
    
    Parameters:
    -----------
    magnitude : float
        Magnitude
    phase : float
        Phase (radians)
        
    Returns:
    --------
    complex
        Complex number
    """
    return magnitude * np.exp(1j * phase)


def self_energy_analytic_continuation(omega_points, self_energy_imag, 
                                    method='pade', n_poles=10):
    """
    Perform analytic continuation of self-energy from imaginary to real axis.
    
    Parameters:
    -----------
    omega_points : array
        Imaginary frequency points (iω_n)
    self_energy_imag : array
        Self-energy on imaginary axis
    method : str
        Continuation method ('pade', 'maximum_entropy')
    n_poles : int
        Number of poles for Pade approximation
        
    Returns:
    --------
    callable
        Function for self-energy on real axis
    """
    if method == 'pade':
        # Pade approximation
        from scipy.optimize import curve_fit
        
        def pade_func(omega, *params):
            # Simple Pade approximation
            n = len(params) // 2
            numerator = sum(params[i] * omega**i for i in range(n))
            denominator = 1 + sum(params[n+i] * omega**i for i in range(1, n))
            return numerator / denominator
        
        # Fit Pade approximation
        p0 = np.zeros(2 * n_poles)
        try:
            popt, _ = curve_fit(pade_func, omega_points.imag, self_energy_imag.real, p0=p0)
            
            def sigma_real(omega_real):
                return pade_func(omega_real, *popt)
                
        except:
            # Fallback to simple interpolation
            from scipy.interpolate import interp1d
            sigma_real = interp1d(omega_points.imag, self_energy_imag.real, 
                                kind='cubic', fill_value='extrapolate')
        
        return sigma_real
    
    else:
        raise NotImplementedError(f"Method {method} not implemented")


def kramers_kronig_transform(omega_array, func_imag):
    """
    Kramers-Kronig transform to get real part from imaginary part.
    
    Re[f(ω)] = (2/π) P ∫₀^∞ ω' Im[f(ω')] / (ω'² - ω²) dω'
    
    Parameters:
    -----------
    omega_array : array
        Frequency array
    func_imag : array
        Imaginary part of function
        
    Returns:
    --------
    array
        Real part of function
    """
    func_real = np.zeros_like(func_imag)
    
    for i, omega in enumerate(omega_array):
        # Principal value integral
        integrand = omega_array * func_imag / (omega_array**2 - omega**2 + 1j*1e-12)
        func_real[i] = (2/np.pi) * np.trapz(integrand.real, omega_array)
    
    return func_real


def hilbert_transform(signal):
    """
    Hilbert transform using FFT.
    
    Parameters:
    -----------
    signal : array
        Input signal
        
    Returns:
    --------
    array
        Hilbert transform of signal
    """
    return special.hilbert(signal)


class NumericalIntegrator:
    """
    Advanced numerical integration tools for multi-dimensional integrals.
    """
    
    @staticmethod
    def adaptive_simpson(func, a, b, tol=1e-8, max_depth=15):
        """
        Adaptive Simpson's rule integration.
        
        Parameters:
        -----------
        func : callable
            Function to integrate
        a, b : float
            Integration limits
        tol : float
            Tolerance
        max_depth : int
            Maximum recursion depth
            
        Returns:
        --------
        float
            Integral value
        """
        def simpson_rule(f, x0, x2):
            x1 = (x0 + x2) / 2
            h = (x2 - x0) / 2
            return h/3 * (f(x0) + 4*f(x1) + f(x2))
        
        def adaptive_simpson_rec(f, a, b, tol, S, fa, fb, fc, depth):
            c = (a + b) / 2
            h = b - a
            d = (a + c) / 2
            e = (c + b) / 2
            fd = f(d)
            fe = f(e)
            
            S_left = h/12 * (fa + 4*fd + fc)
            S_right = h/12 * (fc + 4*fe + fb)
            S2 = S_left + S_right
            
            if depth <= 0 or abs(S2 - S) <= 15 * tol:
                return S2 + (S2 - S) / 15
            
            return (adaptive_simpson_rec(f, a, c, tol/2, S_left, fa, fc, fd, depth-1) +
                   adaptive_simpson_rec(f, c, b, tol/2, S_right, fc, fb, fe, depth-1))
        
        fa = func(a)
        fb = func(b)
        fc = func((a + b) / 2)
        S = simpson_rule(func, a, b)
        
        return adaptive_simpson_rec(func, a, b, tol, S, fa, fb, fc, max_depth)
    
    @staticmethod
    def monte_carlo_2d(func, x_limits, y_limits, n_samples=10000):
        """
        Monte Carlo integration in 2D.
        
        Parameters:
        -----------
        func : callable
            Function to integrate f(x, y)
        x_limits : tuple
            (x_min, x_max)
        y_limits : tuple
            (y_min, y_max)
        n_samples : int
            Number of random samples
            
        Returns:
        --------
        tuple
            (integral_value, error_estimate)
        """
        x_min, x_max = x_limits
        y_min, y_max = y_limits
        
        # Random sampling
        x_samples = np.random.uniform(x_min, x_max, n_samples)
        y_samples = np.random.uniform(y_min, y_max, n_samples)
        
        # Evaluate function
        func_values = np.array([func(x, y) for x, y in zip(x_samples, y_samples)])
        
        # Calculate integral and error
        area = (x_max - x_min) * (y_max - y_min)
        integral = area * np.mean(func_values)
        error = area * np.std(func_values) / np.sqrt(n_samples)
        
        return integral, error