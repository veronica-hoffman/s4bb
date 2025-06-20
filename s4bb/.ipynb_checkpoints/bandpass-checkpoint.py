"""
==========
Bandpasses
==========

Each map is assumed to have an effective bandpass that determines the
amplitude of foreground signals. The bandpass is specified by an array of
frequencies and the relative weight of each frequency.

Some conventions followed here:
* Frequency values are specified in GHz, e.g., nu=100 means 100 GHz. We could
  consider using the astropy.units package (as is used extensively in PySM3),
  but this seems like more trouble than it is worth.
* Bandpass weights are defined as the response per unit frequency to a source
  with uniform spectral radiance, i.e. in units of erg s^{-1} cm^{-2} sr^{-1}
  Hz^{-1}. Note that some experiments, notably WMAP, have used a bandpass
  convention that is response per unit frequency to a source with
  Rayleigh-Jeans spectrum, which is differs by a factor of nu^2.
* Bandpass integrals over frequency are computed using np.trapezoid, which will
  handle irregular binning in frequency.

"""

import numpy as np

# Using np.trapezoid because np.trapz has been deprecated.
# But numpy versions < 2 don't have np.trapezoid. Fix this.
if 'trapezoid' not in dir(np):
    np.trapezoid = np.trapz

# Convert from GHz to Hz
GHz = 1e9
# CMB temperature, in K (Fixsen, ApJ 707, 916, 2009)
Tcmb = 2.72548
# Planck constant, in erg*s
h = 6.62619650e-27
# Boltzmann constant, in erg/K
k = 1.38062259e-16
# Speed of light, in cm/s
c = 2.99792458e10

def delta_Tcmb_to_delta_I(nu):
    x = h * nu * GHz / (k * Tcmb)
    return (2 * k**3 * Tcmb**2 / (h**2 * c**2) * x**4 * np.exp(x) /
            (np.exp(x) - 1)**2)

class Bandpass():
    """
    A Bandpass object represents the response vs frequency of a map (or a
    detector, or many detectors).

    """
    
    def __init__(self, nu, wgt):
        assert len(nu) == len(wgt), "Bandpass nu and wgt inputs must have same length"
        self.nu = nu
        # Normalize weights... not strictly necessary because the
        # bandpass_integral method also includes normalization.
        self.wgt = wgt / np.trapezoid(wgt, x=nu)

    @classmethod
    def deltafn(cls, nu0, epsilon=1e-6):
        nu = np.array([nu0-epsilon, nu0, nu0+epsilon])
        wgt = np.array([0, 1, 0])
        # Return bandpass object
        return cls(nu, wgt)
        
    @classmethod
    def tophat(cls, nu0, nu1, RJ=True, N=100):
        nu = np.linspace(nu0, nu1, N)
        if RJ:
            wgt = nu**(-2)
        else:
            wgt = np.ones(N)
        # Return bandpass object
        return cls(nu, wgt)

    @classmethod
    def from_file(cls, filename, comments='#', delimiter=None, skip_header=0,
                  skip_footer=0, col_nu=0, col_wgt=1):
        finput = np.genfromtxt(filename, comments=comments,
                               delimiter=delimiter, skip_header=skip_header,
                               skip_footer=skip_footer)
        nu = finput[:,col_nu]
        wgt = finput[:,col_wgt]
        # Return bandpass object
        return cls(nu, wgt)

    def fn(self, nu):
        """
        Returns bandpass weights interpolated to requested frequency.

        Parameters
        ----------
        nu : float or array
            Frequency values, in GHz, at which to evaluate bandpass function.

        Returns
        -------
        wgt : float or array
            Bandpass function weight values at the requested frequencies.

        """

        return np.interp(nu, self.nu, self.wgt, left=0.0, right=0.0)
    
    def bandpass_integral(self, fn):
        """
        Calculates bandpass-weighted integral for specified function.

        Parameters
        ----------
        fn : function
            A function that takes an array of frequency values as an input
            and returns an array of values with the same shape.

        Returns
        -------
        int : float
            The bandpass-weighted integral over frequency of the supplied
            function.

        """
        
        return (np.trapezoid(self.wgt * fn(self.nu), x=self.nu) /
                np.trapezoid(self.wgt, x=self.nu))

    def cmb_unit_conversion(self):
        """
        Conversion factor from Delta Tcmb to Delta Intensity calculate for
        this bandpass. Note that this conversion factor has units of
        erg/(s*cm^2*sr*K).

        Parameters
        ----------
        None

        Returns
        -------
        conv : float
            Conversion factor from thermodynamic units (Delta Tcmb) to
            intensity / spectral radiance units.

        """
        
        return self.bandpass_integral(delta_Tcmb_to_delta_I)

    def nu_eff(self, beta=0.0):
        """
        Returns the effective center frequency of the band.

        Parameters
        ----------
        beta : float, optional
            Power weighting to apply to the band when calculating effective
            center frequency. Default value is beta=0, which is flat
            weighting in spectral radiance. You can use other values of beta
            to obtain effective center frequencies for dust or synchrotron
            (but note that power law weighting does not match dust modified
            blackbody).

        Returns
        -------
        nu : float
            Effective center frequency of the band, in GHz.

        """

        return self.bandpass_integral(lambda x: x**(1+beta))

    def to_hdf5(self, fh, group):
        """
        Record bandpass to HDF5 file

        Parameters
        ----------
        fh : h5py File object
            h5py File object should be opened in write mode.
        group : string
            HDF5 group specifier where bandpass arrays will be recorded.

        Returns
        -------
        None

        """

        # Stack the frequency and weight arrays to produce array with
        # shape=(2,N)
        fh[group] = np.stack((self.nu, self.wgt))

    @classmethod
    def from_hdf5(cls, fh, group):
        """
        Read bandpass from HDF5 file

        Parameters
        ----------
        fh : h5py File object
            h5py File object should be opened in read mode.
        group : string
            HDF5 group specifier where bandpass data is located.

        Returns
        -------
        bp : Bandpass object
            Object containing bandpass read in from HDF5 file.

        """

        # Get frequency and weight arrays
        nu = fh[group][0,:]
        wgt = fh[group][1,:]
        # Return bandpass object
        return cls(nu, wgt)
