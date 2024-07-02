import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from scipy.special import gamma

def gamma_dndz(z, z0, beta, d):
    """
    Redshift distribution described in the docs: "https://docs.google.com/presentation/d/1cmspOSFhMx9c2f8EGZT3w-g_eQNDxC01xvrWzGL9ID8/edit#slide=id.g2b173133315_0_12"
    
    Parameters:
    -----------
    z : float or array
        Redshift values at which to evaluate the redshift distribution.
    z0: float
        Pivot redshift parameter.
    beta : float
    	FIXME: see what tha meaning of this parameter
    d: float
    	FIXME: see what the meaning of this parameter
    
    Returns:
    dndz: float or array
    	Redshift distribution evaluated at input redshift z.
    """
    return beta/(gamma(d/beta)*z0**d)*z**(d-1)*np.exp(-(z/z0)**beta)

def srd_dndz(z, z0, alpha):
    """
    Overall redshift distribution described in the
    DESC Science Requirements Document.

    Parameters:
    -----------

    z : float or array
        Redshift values at which to evaluate the redshift distribution.
    z0: float
        Pivot redshift parameter.
    alpha: float
        Power parameter.

    Returns:
    --------
    dndz: float or array.
        Redshift distribution evaluated at input redshift z.
    """
    return z**2*np.exp(-(z/z0)**alpha)


def gaus_kernel(z, sig_z):
    """
    Gaussian kernel to distort photo-z

    Parameters:
    -----------
    z : float or array
        z array of dndz to be distorted (only the dimensions are used)
    sig_z: float or array
        Sigma parameter of Gaussian kernel

    Returns:
    --------
    gk : float or array
        Gaussian kernel centered at 0 with same dimensions as z and sigma = sig_z.
    """
    z_samp = np.linspace(-10*np.max(sig_z), 10*np.max(sig_z), len(z))
    return 1./np.sqrt(2*np.pi*sig_z**2)*np.exp(-0.5*z_samp**2/sig_z**2)


class ZDist(object):
    """
    Base class for redshift distribution
    """
    def __init__(self, z, **kwargs):
        """
        z : float or array
            Redshift values at which to evaluate the redshift distribution.
        """
        self.z = z
        self.Nz = None
        self.zav = None
    

class SpecDESI2LOWZ(ZDist):
    """
    Spec from DESI2 High Density - low-z sample
    follow this docs: "https://docs.google.com/presentation/d/1cmspOSFhMx9c2f8EGZT3w-g_eQNDxC01xvrWzGL9ID8/edit#slide=id.g2b173133315_0_12"
    """
    def __init__(self, z, Nz_center, Nz_width, Nz_nbins, Nz_z0, Nz_beta, Nz_d):
     """
        Parameters:
        -----------
        z : array
            Redshift values at which to evaluate the redshift distribution.
        Nz_nbins : int
            Number of bins/tracers using this type of tracer.
        Nz_ibin : int
            Index of redshift bin considered.
        Nz_z0 : float
            z0 parameter of dndz
        Nz_beta: float
            beta parameter of dndz
        Nz_d: float
       	    d parameter of dndz
     """
     super().__init__(z)
     mask = (self.z > Nz_center- Nz_width / 2) & (self.z < Nz_center + Nz_width / 2)
     dndz_bin = np.zeros_like(self.z)
     dndz_bin[mask] = gamma_dndz(self.z[mask], Nz_z0, Nz_beta, Nz_d)
     self.Nz = dndz_bin
     self.zav = np.average(self.z, weights=self.Nz/np.sum(self.Nz))


     
class LensSRD2018(ZDist):
    """
    Lenses from 2018 SRD
    """
    def __init__(self, z, Nz_center, Nz_width,
                 Nz_nbins, Nz_sigmaz, Nz_alpha=0.94,
                 Nz_z0=0.26, use_filter=True):
        """
        Parameters:
        -----------
        z : float or array
            Redshift values at which to evaluate the redshift distribution.
        Nz_center : float
            Center redshift of N(z).
        Nz_width : float
            Width of redshift slice.
        Nz_nbins : int
            Number of bins/tracers using this type of tracer.
        Nz_sigmaz : float
            Sigma of Gaussian distorting original redshift distribution.
        Nz_alpha : float
            alpha parameter of srd_dndz.
        Nz_z0 : float
            z0 parameter of srd_dndz.
        use_filter : bool
            If `True` it uses scipy.ndimage.gaussian_filter to distort the N(z).
            If `False` it convolves by a Gaussian kernel (as defined in gaus_kernel).
            The defaut (True) reproduces the distributions from the SRD.
        """
        super().__init__(z)
        mask = (self.z > Nz_center - Nz_width / 2) & (self.z < Nz_center + Nz_width / 2)
        dndz_bin = np.zeros_like(self.z)
        dndz_bin[mask] = srd_dndz(self.z[mask], Nz_z0, Nz_alpha)
        # Convolve the SRD N(z) with a Gaussian with the required smearing
        if use_filter:
            dz = z[1] - z[0]
            self.Nz = gaussian_filter(dndz_bin, Nz_sigmaz*(1+Nz_center)/dz)
        else:
            self.Nz = convolve(dndz_bin, gaus_kernel(self.z, Nz_sigmaz*(1+self.z)))
        self.zav = np.average(self.z, weights=self.Nz/np.sum(self.Nz))
    
class SourceSRD2018(ZDist):
    """
    Source from 2018 SRD
    """
    def __init__(self, z, Nz_nbins, Nz_sigmaz, Nz_ibin,
                 Nz_alpha=0.78, Nz_z0=0.13, use_filter=True):
        """
        Parameters:
        -----------
        z : array
            Redshift values at which to evaluate the redshift distribution.
        Nz_nbins : int
            Number of bins/tracers using this type of tracer.
        Nz_sigmaz : float
            Sigma of Gaussian distorting original redshift distribution.
        Nz_ibin : int
            Index of redshift bin considered.
        Nz_alpha : float
            alpha parameter of srd_dndz.
        Nz_z0 : float
            z0 parameter of srd_dndz.
        use_filter : bool
            If `True` it uses scipy.ndimage.gaussian_filter to distort the N(z).
            If `False` it convolves by a Gaussian kernel (as defined in gaus_kernel).
            The defaut (True) reproduces the distributions from the SRD.
        """
        super().__init__(z)
        tile_hi = 1.0 / Nz_nbins * (Nz_ibin + 1)
        tile_low = 1.0 / Nz_nbins * Nz_ibin
        nz_sum = np.cumsum(srd_dndz(self.z, Nz_z0, Nz_alpha))
        nz_sum /= np.sum(srd_dndz(self.z, Nz_z0, Nz_alpha))
        zlow = self.z[np.argmin(np.fabs(nz_sum - tile_low))]
        zhi = self.z[np.argmin(np.fabs(nz_sum - tile_hi))]
        mask = (self.z >= zlow) & (self.z <= zhi)
        dndz_bin = np.zeros_like(self.z)
        dndz_bin[mask] = srd_dndz(self.z[mask], Nz_z0, Nz_alpha)
        if use_filter:
            zcent = 0.2 * Nz_ibin + 0.25  # Start at z=0.2 -- Caution!! this is hacky
            dz = self.z[1] - self.z[0]
            self.Nz = gaussian_filter(dndz_bin, Nz_sigmaz*(1+zcent)/dz)
        else:
            self.Nz = convolve(dndz_bin, gaus_kernel(self.z, Nz_sigmaz*(1+self.z)))
        self.zav = np.average(self.z, weights=self.Nz/np.sum(self.Nz))


class TopHat(ZDist):
    """
    Top Hat redshift bin
    """
    def __init__(self, z, Nz_center, Nz_width):
        """
        Parameters:
        -----------
        z : array
            Redshift values at which to evaluate the redshift distribution.
        Nz_center : float
            Center of the redshift bin.
        Nz_width : float
            Width of top-hat bin.
        """
        super().__init__(z)
        self.Nz = np.zeros_like(z)
        mask = (self.z >= Nz_center - Nz_width/2) & (self.z <= Nz_center + Nz_width/2)
        self.Nz[mask] = 1.
        self.Nz /= np.sum(self.Nz)


class Gaussian(ZDist):
    """
    Gaussian redshift bin
    """
    def __init__(self, z, Nz_mu, Nz_sigma):
        """
        Parameters:
        -----------
        z : array
            Redshift values at which to evaluate the redshift distribution.
        Nz_mu : float
            Mean of the redshift distribution.
        Nz_sigma : float
            Sigma of the redshift distribution.
        """
        super().__init__(z)
        self.Nz = np.exp(-0.5*(self.z - Nz_mu)**2/Nz_sigma**2)


class WLSource(object):
    """
    Weak lensing source tracer
    """
    def __init__(self, sacc_tracer, ellipticity_error, number_density,
                 zdist, mult_bias=0, ia_bias=0, **ia_kwargs):
        """
        Parameters:
        -----------
        sacc_tracer : str
            Name of sacc tracer to consider.
        ellipticity_error : float
            Ellipticity error of the tracer. This value gets propagated.
        number_density : float
            Effective number density.
        zdist : ZDist
            Redshift distribution
        """
        self.sacc_tracer = sacc_tracer
        self.ellipticity_error = ellipticity_error
        self.number_density = number_density
        self.zdist = zdist
        self.mult_bias = mult_bias
        self.ia_bias = ia_bias
        self.ia_kwargs = ia_kwargs

    def to_dict(self):
        """
        Write WLSource object to a dictionary that can be added
        to a sacc file.
        """
        dict_out = self.__dict__
        dict_out['kind'] = 'WLSource'
        return dict_out
