import numpy as np


def get_noise_power(config, src):
    """ Returns noise power for tracer

    Parameters:
    ----------
    config : dict
        The dictinary containt the relevant two_point section of the config

    src : str
        Tracer ID

    Returns:
    -------
    noise : float
       Noise power for Cls for that particular tracer. That is 1/nbar for
       number counts and e**2/nbar for weak lensing tracer.

    Note:
    -----
    The input number_densities are #/arcmin.
    The output units are in steradian.
    """

    d = config["sources"][src]
    nbar = d["number_density"] * (180 * 60 / np.pi) ** 2  # per steradian
    kind = d["kind"]
    if kind == "WLSource":
        noise_power = d["ellipticity_error"] ** 2 / nbar
    elif kind == "NumberCountsSource":
        noise_power = 1 / nbar
    else:
        print("Cannot do error for source of kind %s." % (kind))
        raise NotImplementedError
    return noise_power


def gauss_cov(config, sacc_data, preds, add_noise=False):
    """
    Parameters:
    -----------
    config : dict
        Input configuration
    sacc_data: Sacc
        Input sacc data
    """
    covar = np.zeros(len(sacc_data.data))
    for name, scfg in config["statistics"].items():
        # now add errors -- this is a quick hack
        if "xi" in scfg["sacc_data_type"]:
            print("Sorry, cannot yet do errors for correlation function")
            raise NotImplementedError
        else:
            ell_edges = np.array(scfg["ell_edges"])
            # note: sum(2l+1,lmin..lmax) = (lmax+1)^2-lmin^2
            # Nmodes = config["fsky"] * ((ell_edges[1:] + 1) ** 2
            #                           - (ell_edges[:-1]) ** 2)
            ells = 0.5*(ell_edges[1:]+ell_edges[:-1])
            delta_ell = np.gradient(ells)
            norm = config["fsky"]*(2*ells+1)*delta_ell
            # noise power
            # now find the two sources and their noise powers
            # the auto powers -- this should work equally well for auto and
            # cross
            auto1, auto2 = [
                preds[(src, src)][0] + get_noise_power(config, src)
                for src in scfg["sources"]
            ]
            for src in scfg["sources"]:
                print(src, get_noise_power(config, src))
            max_len = np.min([len(auto1), len(auto2)])
            cross, ndx = preds[tuple(scfg["sources"])]
            max_len = np.min([max_len, len(cross)])
            var = (auto1[:max_len] * auto2[:max_len] +
                   cross[:max_len] * cross[:max_len]) / norm[:max_len]
            covar[ndx] = var
            for n, err in zip(ndx, np.sqrt(var)):
                sacc_data.data[n].error = np.sqrt(err)
                if add_noise:
                    sacc_data.data[n].value += np.random.normal(0, err)
    return covar, sacc_data
