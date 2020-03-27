#
# Data generation module
#

import firecrown
import sacc
import copy
import numpy as np


def generate(config):
    """ Generates data using dictionary based config.
    """

    verbose = config['verbose']
    gen_config = config['generate']
    gen_config['verbose'] = verbose

    capabilities = [('two_point', two_point_template, two_point_insert)]
    sacc_outputs = {}

    for name, gen_template, _ in capabilities:
        if name in gen_config:
            if verbose:
                print("Generating %s template..." % name)
            gen_config[name]['verbose'] = verbose
            gen_template(gen_config[name])

    if verbose:
        print("Stoking the bird for predictions...")
    #print (gen_config)
    config, data = firecrown.parse(firecrown_sanitize(gen_config))
    cosmo = firecrown.get_ccl_cosmology(config['parameters'])
    firecrown.compute_loglike(cosmo=cosmo, data=data)

    for name, _, gen_insert in capabilities:
        if verbose:
            print("Writing %s data..." % name)
        gen_insert(gen_config[name], data)


def firecrown_sanitize(config):
    """ Sanitizes the input for firecrown, that is removes keys that firecrown doesn't recognize
    """
         

    def delkeys(dic, keys):
        for k in keys:
            if k in dic:
                del dic[k]

    # removes keys that firecrown hates
    fconfig = copy.copy(config)  # we want a shallow copy!
    del fconfig['verbose']
    for tcfg in fconfig['two_point']['sources'].values():
        delkeys(tcfg, ['Nz_type', 'Nz_center', 'Nz_width', 'Nz_sigma'])
    for tcfg in fconfig['two_point']['statistics'].values():
        delkeys(tcfg, ['ell', 'theta'])

    return fconfig


def two_point_template(config):
    S = sacc.Sacc()
    verbose = config['verbose']
    if verbose:
        print("Generating tracers: ", end="")
    for src, tcfg in config['sources'].items():
        if verbose:
            print(src, " ", end="")
        if "Nz_type" not in tcfg:
            print("Missing Nz_type in %s. Quitting." % src)
            stop()
        if tcfg['Nz_type'] == 'Gaussian':
            mu, sig = tcfg['Nz_center'], tcfg['Nz_sigma']
            zar = np.linspace(max(0, mu-5*sig), mu+5*sig, 500)
            Nz = np.exp(-(zar-mu)**2/(2*sig**2))
            S.add_tracer('NZ', src, zar, Nz)
        elif tcfg['Nz_type'] == 'TopHat':
            mu, wi = tcfg['Nz_center'], tcfg['Nz_width']
            zar = np.linspace(max(0, mu-wi/2), mu+wi/2, 5)
            Nz = np.ones(5)
            S.add_tracer('NZ', src, zar, Nz)
        else:
            print("Bad Nz_type in %s. Quitting." % src)
            stop()
    if verbose:
        print("\nGenerating data slots: ", end="")
    for name, scfg in config['statistics'].items():
        print(name, " ", end="")
        dt = scfg['sacc_data_type']
        src1, src2 = scfg['sources']
        if 'cl' in dt:
            ells = scfg['ell']
            for ell in ells:
                S.add_data_point(dt, (src1, src2), 0.0, ell=ell, error=1e30)
        elif 'xi' in dt:
            thetas = scfg['theta']
            for theta in thetas:
                S.add_data_point(dt, (src1, src2), 0.0,
                                 theat=theta, error=1e30)
        else:
            print("Cannot process %s. Quitting." % dt)
            stop()

    if verbose:
        print()
    config['sacc_data'] = S


def two_point_insert(config, data):
    verbose = config['verbose']
    sacc = config['sacc_data']
    for name, scfg in config['statistics'].items():
        pred = data['two_point']['data']['statistics'][name].predicted_statistic_
        # identify data points in sacc
        ndx = []
        for i, d in enumerate(sacc.data):
            if ((d.data_type == scfg['sacc_data_type']) and
                d.tracers[0] == scfg['sources'][0] and
                    d.tracers[1] == scfg['sources'][1]):
                ndx.append(i)
        assert(len(ndx) == len(pred))
        for n, p in zip(ndx, pred):
            sacc.data[n].value = p
    if 'sacc_file' in config:
        if verbose:
            print("Writing %s ..." % config['sacc_file'])
        sacc.save_fits(config['sacc_file'], overwrite=True)
