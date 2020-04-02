#
# Data generation module
#

import firecrown
import sacc
import copy
import numpy as np


def generate(config):
    """ Generates data using dictionary based config.

    Parameters:
    ----------
    config : dict
        The yaml parsed dictional of the input yaml file

    """

    verbose = config['verbose']
    gen_config = config['generate']
    gen_config['verbose'] = verbose

    capabilities = [
        ('two_point',
         'firecrown.ccl.two_point',
         two_point_template,
         two_point_insert)]
    process = []

    for dname, ddict in gen_config.items():
        if (not hasattr(ddict, "__getitem__") or
                "module" not in ddict):
            continue  # clearly not our banana
        for name, moduleid, gen_template, gen_insert in capabilities:
            if ddict['module'] == moduleid:
                if verbose:
                    print(
                        "Generating %s template for section %s..." %
                        (name, dname))
                process.append((dname, name, gen_insert))
                gen_config[dname]['verbose'] = verbose
                gen_template(gen_config[dname])
            continue

    if verbose:
        print("Stoking the bird for predictions...")

    config, data = firecrown.parse(firecrown_sanitize(gen_config))
    cosmo = firecrown.get_ccl_cosmology(config['parameters'])
    firecrown.compute_loglike(cosmo=cosmo, data=data)

    for dname, name, gen_insert in process:
        if verbose:
            print("Writing %s data for section %s..." % (name, dname))
        gen_insert(gen_config[dname], data)


def firecrown_sanitize(config):
    """ Sanitizes the input for firecrown, that is removes keys that firecrown
        doesn't recognize

    Parameters:
    ----------
    config : dict
        The dictinary that is ready for firecrown to munch on, but might have
        extra keys that are augur specific

    Returns:
    -------
    config : dict
       The input dictionary with unwatned keys removed

    """

    def delkeys(dic, keys):
        for k in keys:
            if k in dic:
                del dic[k]

    # removes keys that firecrown hates
    fconfig = copy.deepcopy(config)
    delkeys(fconfig, ['verbose', 'augur'])
    for tcfg in fconfig['two_point']['sources'].values():
        delkeys(tcfg, ['Nz_type', 'Nz_center', 'Nz_width', 'Nz_sigma',
                       'ellipticity_error', 'number_density'])
    for tcfg in fconfig['two_point']['statistics'].values():
        delkeys(tcfg, ['ell_edges', 'theta_edges'])

    return fconfig


def two_point_template(config):
    """Creates a template SACC object with tracers and statistics

    Parameters:
    ----------
    config : dict
        The dictinary containt the relevant two_point section of the config

    Returns:
    -------
    sacc : sacc object

       Sacc objects with appropriate tracers and measurement slots
       (i.e. data vectors with associated correlation functions,
       angles, etc), but with zeros for measurement values

    """

    S = sacc.Sacc()
    verbose = config['verbose']
    if verbose:
        print("Generating tracers: ", end="")
    for src, tcfg in config['sources'].items():
        if verbose:
            print(src, " ", end="")
        if "Nz_type" not in tcfg:
            print("Missing Nz_type in %s. Quitting." % src)
            raise RuntimeError
        if tcfg['Nz_type'] == 'Gaussian':
            mu, sig = tcfg['Nz_center'], tcfg['Nz_sigma']
            zar = np.linspace(max(0, mu - 5 * sig), mu + 5 * sig, 500)
            Nz = np.exp(-(zar - mu)**2 / (2 * sig**2))
            S.add_tracer('NZ', src, zar, Nz)
        elif tcfg['Nz_type'] == 'TopHat':
            mu, wi = tcfg['Nz_center'], tcfg['Nz_width']
            zar = np.linspace(max(0, mu - wi / 2), mu + wi / 2, 5)
            Nz = np.ones(5)
            S.add_tracer('NZ', src, zar, Nz)
        else:
            print("Bad Nz_type in %s. Quitting." % src)
            raise RuntimeError
    if verbose:
        print("\nGenerating data slots: ", end="")
    for name, scfg in config['statistics'].items():
        if verbose:
            print(name, " ", end="")
        dt = scfg['sacc_data_type']
        src1, src2 = scfg['sources']
        if 'cl' in dt:
            ell_edges = np.array(scfg['ell_edges'])
            ells = 0.5 * (ell_edges[:-1] + ell_edges[1:])
            for ell in ells:
                S.add_data_point(dt, (src1, src2), 0.0, ell=ell, error=1e30)
        elif 'xi' in dt:
            theta_edges = np.array(scfg['theta_edges'])
            thetas = 0.5 * (theta_edges[:-1] + theta_edges[1:])
            for theta in thetas:
                S.add_data_point(dt, (src1, src2), 0.0,
                                 theta=theta, error=1e30)
        else:
            print("Cannot process %s. Quitting." % dt)
            raise NotImplementedError

    if verbose:
        print()
    config['sacc_data'] = S


def getNoisePower(config, src):
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
       number counts and e**2/nbar for weak lensing tracer

    """

    d = config['sources'][src]
    nbar = d['number_density'] * (180 * 60 / np.pi)**2  # per steradian
    kind = d['kind']
    if kind == 'WLSource':
        NPower = d['ellipticity_error']**2 / nbar
    elif kind == 'NumberCountsSource':
        NPower = 1 / nbar
    else:
        print("Cannot do error for source of kind %s." % (kind))
    return NPower


def two_point_insert(config, data):
    """ Copies the firecrown caclulate theory back into the sacc object
        and adds noise (as a hack before we get TJPCov running) and then
        saves sacc to disk.

    Parameters:
    ----------
    config : dict
        The dictinary containt the relevant two_point section of the config
    data : dict
        The firecrown's data strcuture where data predictions are stored

    """

    verbose = config['verbose']
    sacc = config['sacc_data']
    add_noise = config['add_noise']
    preds = {}
    if verbose:
        print("Writing predictions back into sacc...")
    stats = data['two_point']['data']['statistics']
    for name, scfg in config['statistics'].items():
        pred = stats[name].predicted_statistic_
        # identify data points in sacc
        ndx = []
        for i, d in enumerate(sacc.data):
            if ((d.data_type == scfg['sacc_data_type']) and
                d.tracers[0] == scfg['sources'][0] and
                    d.tracers[1] == scfg['sources'][1]):
                ndx.append(i)
        assert(len(ndx) == len(pred))
        preds[tuple(scfg['sources'])] = (
            pred, ndx)  # predicted power for noise

        for n, p in zip(ndx, pred):
            sacc.data[n].value = p

    if verbose:
        print("Adding errors...")
    # now add errors, this is very fragile as it
    # assumes the same binning everywhere, etc. Needs to be replaced by TJPCov
    # asap
    covar = np.zeros(len(sacc.data))
    for name, scfg in config['statistics'].items():
        # now add errors -- this is a quick hack
        if "xi" in scfg['sacc_data_type']:
            print("Sorry, cannot yet do errors for correlation function")
            raise NotImplementedError
        else:
            ell_edges = np.array(scfg['ell_edges'])
            # note: sum(2l+1,lmin..lmax) = (lmax+1)^2-lmin^2
            Nmodes = config['fsky'] * \
                ((ell_edges[1:] + 1)**2 - (ell_edges[:-1])**2)
            # noise power
            # now find the two sources and their noise powers
            # the auto powers -- this should work equally well for auto and
            # cross
            auto1, auto2 = [preds[(src, src)][0] + getNoisePower(config, src)
                            for src in scfg['sources']]
            cross, ndx = preds[tuple(scfg['sources'])]
            var = (auto1 * auto2 + cross * cross) / Nmodes
            covar[ndx] = var
            for n, err in zip(ndx, np.sqrt(var)):
                sacc.data[n].error = np.sqrt(err)
                if add_noise:
                    sacc.data[n].value += np.random.normal(0, err)

    assert(np.all(covar > 0))
    # because firecrown only supports FullCovariance
    sacc.add_covariance(np.diag(covar))
    if 'sacc_file' in config:
        if verbose:
            print("Writing %s ..." % config['sacc_file'])
        sacc.save_fits(config['sacc_file'], overwrite=True)
