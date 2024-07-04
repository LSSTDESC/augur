import numpy as np
import math
from scipy.ndimage import gaussian_filter, convolve


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
    return (z/z0)**2.*np.exp(-(z/z0)**alpha)


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


def equal_density_zbins(z, nz, nbins):
    """
    This function takes a redshift distribution and returns the redshift bins
    that contain equal number of galaxies.

    Parameters:
    -----------
    z : array
        Redshift values at which to evaluate the redshift distribution.
    nz : array
        Redshift distribution.
    nbins : int
        Number of redshift bins.
    """
    # based on N. Tessore's cumtrapz method implemented in Glass
    cuml_nz = np.empty_like(nz)
    np.cumsum((nz[..., 1:] + nz[..., :-1])/2*np.diff(z), axis=-1, out=cuml_nz[..., 1:])
    cuml_nz[..., 0] = 0
    cuml_nz /= cuml_nz[-1]
    zbinedges = np.interp(np.linspace(0, 1, nbins+1), cuml_nz, z)
    return list(zip(zbinedges, zbinedges[1:]))


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
    Source from 2018 SRD, benchmarked against Paul Rogozenski's notebook
    """
    def __init__(self, z, Nz_nbins, Nz_sigmaz, Nz_ibin,
                 Nz_alpha=0.78, Nz_z0=0.13):
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
        """
        super().__init__(z)
        dndz = srd_dndz(self.z, Nz_z0, Nz_alpha)
        zbins = equal_density_zbins(self.z, dndz, Nz_nbins)
        # Based on A. Loureiro's implementation in Glass
        zbins_arr = np.asanyarray(zbins)
        z_high = zbins_arr[:, 1, np.newaxis]
        z_low = zbins_arr[:, 0, np.newaxis]
        # vectorises the erf function:

        erf_vec = np.vectorize(math.erf, otypes=(float,))
        sz = 2 ** 0.5 * Nz_sigmaz * (1 + self.z)
        binned_nz = erf_vec((z - z_low)/sz)
        binned_nz -= erf_vec((z - z_high)/sz)
        binned_nz /= 1 + erf_vec(z / sz)
        binned_nz *= dndz
        # FIXME: this function is vectorised but the generate.py loop is not
        # FIXME: so I am generating all the bins but only using one at the time
        self.Nz = binned_nz[Nz_ibin]
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


class ZDistFromFile(ZDist):
    """
    Class to pass an arbitrary redshift distribution to a tracer
    """
    def __init__(self, input_file, ibin=None, format='npy', **kwargs):
        """
        Parameters:
        -----------
        input_file : str
            Path to input file with z and dndz
        """
        if 'npy' in format:
            data = np.load(input_file, allow_pickle=True)
            self.z = data[()]['redshift_range']
            if ibin is None:
                raise ValueError('Expected bin number for this format')
            else:
                self.Nz = data[()]['bins'][ibin]
        elif 'ascii' in format:
            data = np.loadtxt(input_file)
            self.z = data[:, 0]
            if ibin is None:
                self.Nz = data[:, 1]
            else:
                self.Nz = data[:, ibin+1]
        else:
            raise NotImplementedError('Only ascii files or npy files are currently supported')

        self.zav = np.average(self.z, weights=self.Nz/np.sum(self.Nz))
