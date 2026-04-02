# Augur Configuration Guide

This document describes the structure and options of the Augur YAML configuration
file used to set up Fisher matrix forecasts for LSST DESC two-point analyses.

## Overview

Augur reads a single YAML file that defines everything needed to:

1. **Generate** a fiducial data vector and covariance matrix (stored as a SACC file).
2. **Analyze** the data vector via Fisher matrix forecasting.
3. **Postprocess** results into triangle plots and LaTeX tables.

### Jinja2 Templating

The config file supports **Jinja2 template expressions**. This lets you reference
environment variables anywhere:

```yaml
input_file: "{{ env['AUGUR_DIR'] }}/data/srd_source_bins_y1.txt"
```

Make sure the relevant environment variables (e.g. `AUGUR_DIR`) are set before
running Augur.

### Running Augur

From the command line:

```bash
augur config.yml [-v]
```

Or via Python:

```python
from augur.generate import generate
from augur.analyze import Analyze
from augur.postprocess import postprocess

# Generate fiducial data vector
lk = generate('config.yml', return_all_outputs=False)

# Run Fisher analysis
ao = Analyze('config.yml')
ao.get_fisher_bias(method='5pt_stencil')

# Postprocess results
postprocess('config.yml')
```

---

## Configuration Sections

### `general`

Top-level flags controlling analysis behavior.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ignore_scale_cuts` | `bool` | `False` | If `True`, ignores all `kmax`/`lmax` values when building the data vector—all multipoles are included. |
| `ignore_scale_cuts_likelihood` | `bool` | `False` | If `True`, ignores scale cuts when constructing the Firecrown likelihood. Cannot be `True` if `ignore_scale_cuts` is `False`. |
| `bandpower_windows` | `str` or `None` | `None` | Controls bandpower window treatment. Options: `'tophat'` (adds top-hat bandpower windows to SACC), `'NaMaster'` (not yet implemented), or `None` (no windows). |

```yaml
general:
    ignore_scale_cuts: False
```

---

### `cosmo`

Defines the fiducial cosmology. All keys are passed directly to `pyccl.Cosmology(...)`.

| Key | Type | Description |
|-----|------|-------------|
| `Omega_c` | `float` | Cold dark matter density parameter. |
| `Omega_b` | `float` | Baryon density parameter. |
| `h` | `float` | Dimensionless Hubble parameter ($H_0 / 100$). |
| `n_s` | `float` | Scalar spectral index. |
| `sigma8` | `float` | RMS matter fluctuation in 8 $h^{-1}$ Mpc spheres. **Mutually exclusive** with `A_s`. |
| `A_s` | `float` | Scalar amplitude of primordial power spectrum. **Mutually exclusive** with `sigma8`. |
| `w0` | `float` | Dark energy equation of state parameter $w_0$. |
| `wa` | `float` | Dark energy equation of state time derivative $w_a$. |
| `Omega_k` | `float` | Curvature density parameter. |
| `m_nu` | `float` | Sum of neutrino masses in eV. |
| `transfer_function` | `str` | Transfer function. Options: `'boltzmann_camb'` (default), `'boltzmann_class'`, `'eisenstein_hu'`, `'bbks'`. |
| `matter_power_spectrum` | `str` | Nonlinear matter power spectrum method (e.g. `'halofit'`). |
| `extra_parameters` | `dict` | Nested dictionary for backend-specific settings. |

#### CAMB-specific extra parameters

```yaml
cosmo:
    extra_parameters:
        camb:
            dark_energy_model: 'ppf'
            halofit_version: 'takahashi'
```

Supported `halofit_version` values that trigger HMCode baryon feedback:
`'mead2020_feedback'`, `'mead'`, `'mead2015'`, `'mead2016'`.

#### Modified gravity

Augur supports the $\mu$–$\Sigma$ parametrization for modified gravity via
`pyccl`. Include the `mg_parametrization` sub-key under `cosmo` and set the
transfer function to `boltzmann_isitgr`:

```yaml
cosmo:
    mg_parametrization:
        mu_Sigma:
            mu_0: 0.0
            sigma_0: 0.0
            c1_mg: 1.0
            c2_mg: 1.0
            lambda_mg: 0.0
    transfer_function: boltzmann_isitgr
    matter_power_spectrum: linear
```

The corresponding Fisher parameters are named `mg_musigma_mu`, `mg_musigma_sigma`,
`mg_musigma_c1`, `mg_musigma_c2`, and `mg_musigma_lambda0`.
See [srd_y1_3x2_mg.yml](srd_y1_3x2_mg.yml) for a complete example.

---

### `systematics`

Defines **fiducial values** of all nuisance/systematic parameters. These are
passed to the Firecrown likelihood as a `ParamsMap` and serve as the pivot point
for Fisher derivatives.

#### Source (weak lensing) systematics

| Key pattern | Description |
|-------------|-------------|
| `src{i}_mult_bias` | Multiplicative shear bias for source bin `i` (typically `0.0`). |
| `src{i}_delta_z` | Photo-z shift for source bin `i` (typically `0.0`). |

#### Intrinsic alignment parameters

| Key | Description |
|-----|-------------|
| `ia_bias` | IA amplitude $A_{\mathrm{IA}}$ (NLA model). |
| `alphaz` | IA redshift power-law slope $\alpha_z$. |
| `z_piv` | IA pivot redshift $z_{\mathrm{piv}}$. |

#### Lens (number counts) systematics

| Key pattern | Description |
|-------------|-------------|
| `lens{i}_bias` | Linear galaxy bias for lens bin `i`. |
| `lens{i}_alphaz` | Lens bias redshift evolution slope. |
| `lens{i}_z_piv` | Lens bias pivot redshift. |
| `lens{i}_alphag` | Lens bias magnitude slope. |
| `lens{i}_delta_z` | Photo-z shift for lens bin `i`. |
| `lens{i}_sigma_z` | Photo-z stretch factor for lens bin `i` (typically `1.0`). |

```yaml
systematics:
    src0_mult_bias: 0.0
    src0_delta_z: 0.0
    ia_bias: 5.717
    alphaz: -0.47
    z_piv: 0.3
    lens0_bias: 1.562362
    lens0_delta_z: 0.0
    lens0_sigma_z: 1.0
```

---

### `Firecrown_Factory`

Configures the Firecrown likelihood using the factory pattern. When present,
Augur builds the likelihood from factory objects instead of manually assembling
two-point statistics.

```yaml
Firecrown_Factory:
    TwoPointFactory:
        correlation_space: harmonic
        number_counts_factories:
            - type_source: default
              global_systematics: []
              include_rsd: false
              per_bin_systematics:
                - {type: PhotoZShiftandStretchFactory}
                - {type: LinearBiasSystematicFactory}
        weak_lensing_factories:
            - type_source: default
              global_systematics:
                - {type: LinearAlignmentSystematicFactory, alphag: 1}
              per_bin_systematics:
                - {type: MultiplicativeShearBiasFactory}
                - {type: PhotoZShiftFactory}
        cmb_factories: []
        int_options: null
```

#### Top-level keys

| Key | Description |
|-----|-------------|
| `correlation_space` | `'harmonic'` (Fourier / $C_\ell$ space). |
| `number_counts_factories` | List of factory configs for galaxy number counts probes. |
| `weak_lensing_factories` | List of factory configs for weak lensing probes. |
| `cmb_factories` | List of CMB lensing factories (currently `[]`). |
| `int_options` | Integration options (`null` for defaults). |

#### Per-factory keys

| Key | Description |
|-----|-------------|
| `type_source` | Source type: `'default'`. |
| `global_systematics` | List of systematic factories shared across all bins. |
| `per_bin_systematics` | List of systematic factories applied per bin. |
| `include_rsd` | (number counts only) Include redshift-space distortions. |

#### Available systematic factory types

| Type | Use for | Notes |
|------|---------|-------|
| `PhotoZShiftFactory` | Source photo-z shifts | |
| `PhotoZShiftandStretchFactory` | Lens photo-z shift + stretch | |
| `MultiplicativeShearBiasFactory` | Source multiplicative shear bias | |
| `LinearBiasSystematicFactory` | Lens linear galaxy bias | |
| `LinearAlignmentSystematicFactory` | NLA intrinsic alignment | Accepts `alphag` parameter |
| `TattAlignmentSystematicFactory` | TATT IA model | Accepts `include_z_dependence` |

---

### `ccl_accuracy`

Fine-tunes numerical accuracy parameters in `pyccl`.

```yaml
ccl_accuracy:
    spline_params:
        K_MAX_SPLINE: 100
    gsl_params:
        INTEGRATION_EPSREL: 1e-6
        INTEGRATION_LIMBER_EPSREL: 1e-2
```

| Sub-section | Description |
|-------------|-------------|
| `spline_params` | Overrides for `pyccl.spline_params` (e.g. `K_MAX_SPLINE`). |
| `gsl_params` | Overrides for `pyccl.gsl_params` (e.g. `INTEGRATION_EPSREL`, `INTEGRATION_LIMBER_EPSREL`). |

---

### `pt_calculator`

Configures a perturbation theory calculator passed to Firecrown's `ModelingTools`.
If omitted, no PT calculator is used.

```yaml
pt_calculator:
    type: eulerian_pt_calculator
    with_NC: False
    with_IA: True
    log10k_min: -4
    log10k_max: 2
    nk_per_decade: 80
    with_matter_1loop: False
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `type` | `str` | `'eulerian_pt_calculator'` | Calculator type. Options: `'eulerian_pt_calculator'`, `'lagrangian_pt_calculator'`, `'bacco_lbias_calculator'`. |

All other keys are passed directly to the corresponding `pyccl.nl_pt` constructor.

---

### `hm_calculator`

Configures a halo model calculator passed to Firecrown's `ModelingTools`.
If omitted, no halo model calculator is used. All keys are passed directly to
`pyccl.hm.HaloModelCalculator(...)`.

```yaml
hm_calculator:
    mass_function: 'tinker10'
    halo_bias: 'tinker10'
```

---

### `sources`

Configures **weak lensing source (shear) tracers**.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `nbins` | `int` | Yes | Number of source tomographic bins. Tracers are named `src0` … `src{nbins-1}`. |
| `ndens` | `float` or `list` | Yes | Galaxy number density in arcmin⁻². Scalar = total across bins; list = per-bin. |
| `ellipticity_error` | `float` | Yes | RMS ellipticity error ($\sigma_e$). Enters shape noise: $N_\ell = \sigma_e^2 / \bar{n}$. |
| `Nz_type` | `str` or `list[str]` | Yes | Redshift distribution class. See below. |
| `Nz_kwargs` | `dict` | Yes | Keyword arguments for the `Nz_type` constructor. |

#### Additional legacy-path keys (optional)

| Key | Type | Description |
|-----|------|-------------|
| `mult_bias` | `float` or `list` | Multiplicative shear bias. Scalar or per-bin list. |
| `delta_z` | `float` or `list` | Photo-z shift. Scalar or per-bin list. |
| `ia_class` | `str` | IA systematic class, e.g. `'wl.LinearAlignmentSystematic'`. |
| `ia_bias` | `float` | IA amplitude. |
| `alphaz` | `float` | IA redshift slope. |
| `z_piv` | `float` | IA pivot redshift. |

#### Available `Nz_type` options

| Type | Description | Required `Nz_kwargs` |
|------|-------------|---------------------|
| `ZDistFromFile` | Load $n(z)$ from a text or numpy file. | `input_file`, `format` (`'ascii'` or `'npy'`) |
| `SourceSRD2018` | Analytic SRD 2018 source $n(z)$. | `Nz_alpha`, `Nz_z0`, `Nz_sigmaz` |
| `LensSRD2018` | Analytic SRD 2018 lens $n(z)$. | `Nz_width`, `Nz_center`, `Nz_sigmaz`, `Nz_alpha`, `Nz_z0` |
| `TopHat` | Top-hat (uniform) redshift distribution. | `Nz_center`, `Nz_width` |
| `Gaussian` | Gaussian redshift distribution. | `Nz_mu`, `Nz_sigma` |

```yaml
sources:
    nbins: 5
    ndens: 10
    ellipticity_error: 0.26
    Nz_type: 'ZDistFromFile'
    Nz_kwargs:
        input_file: "{{ env['AUGUR_DIR'] }}/data/srd_source_bins_y1.txt"
        format: 'ascii'
```

---

### `lenses`

Configures **galaxy number counts (lens) tracers**. Structure mirrors `sources`.

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `nbins` | `int` | Yes | Number of lens tomographic bins. Tracers named `lens0` … `lens{nbins-1}`. |
| `ndens` | `float` or `list` | Yes | Number density in arcmin⁻². |
| `delta_z` | `float` | No | Global photo-z shift for lenses (default `0`). |
| `Nz_type` | `str` or `list[str]` | Yes | Same options as `sources.Nz_type`. |
| `Nz_kwargs` | `dict` | Yes | Same pattern as `sources.Nz_kwargs`. |
| `bias_type` | `str` | No | Galaxy bias model: `'inverse_growth'` or `'custom'`. Legacy path only. |
| `bias_kwargs` | `dict` | No | For `'custom'`: `b: [list]`. For `'inverse_growth'`: `b0: float`. |

```yaml
lenses:
    nbins: 5
    ndens: 18
    Nz_type: 'ZDistFromFile'
    Nz_kwargs:
        input_file: "{{ env['AUGUR_DIR'] }}/data/srd_lens_bins_y1.txt"
        format: 'ascii'
    bias_type: 'custom'
    bias_kwargs:
        b: [1.562362, 1.732963, 1.913252, 2.100644, 2.293210]
```

---

### `statistics`

Defines which two-point statistics to include in the data vector. Each key is a
SACC-style statistic type.

| Statistic key | Physical observable | Tracer pair convention |
|---------------|--------------------|-----------------------|
| `galaxy_density_cl` | Galaxy clustering $C_\ell^{gg}$ | `[lens_i, lens_j]` |
| `galaxy_shear_cl_ee` | Cosmic shear $C_\ell^{\epsilon\epsilon}$ | `[src_i, src_j]` |
| `galaxy_shearDensity_cl_e` | Galaxy-galaxy lensing $C_\ell^{g\epsilon}$ | `[lens_i, src_j]` |

#### Per-statistic keys

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `tracer_combs` | `list[list[int]]` | Yes | List of bin-index pairs for this statistic. |
| `ell_edges` | `str` (evaluable) | Yes | Python expression for $\ell$-bin edges (e.g. `np.geomspace(20, 15000, 21, endpoint=True)`). Band centers are geometric means: $\ell_c = \sqrt{\ell_{\mathrm{low}} \cdot \ell_{\mathrm{high}}}$. |
| `kmax` | `float` or `None` | No | Maximum wavenumber cut in Mpc⁻¹. Converted to $\ell_{\max}$ via $\ell_{\max} = k_{\max} \cdot \chi(\bar{z})$. **Mutually exclusive** with `lmax`. |
| `lmax` | `float`, `list`, or `None` | No | Direct maximum multipole cut. Scalar (all combos) or list (per-combo). **Mutually exclusive** with `kmax`. |

```yaml
statistics:
    galaxy_density_cl:
        tracer_combs: [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        ell_edges: np.geomspace(20, 15000, 21, endpoint=True)
        kmax: 0.201
    galaxy_shear_cl_ee:
        tracer_combs: [[0, 0], [0, 1], [1, 1]]  # etc.
        ell_edges: np.geomspace(20, 15000, 21, endpoint=True)
        kmax: null
    galaxy_shearDensity_cl_e:
        tracer_combs: [[0, 2], [0, 3], [1, 3]]  # [lens, source]
        ell_edges: np.geomspace(20, 15000, 21, endpoint=True)
        kmax: 0.201
```

---

### `fiducial_sacc_path`

Output path for the generated SACC file containing the fiducial data vector and
covariance matrix. Supports Jinja2 env vars.

```yaml
fiducial_sacc_path: test_sacc.sacc
```

---

### `cov_options`

Controls covariance matrix generation. The `cov_type` key selects the method.

#### Option 1: `gaus_internal` — Analytic Gaussian covariance

```yaml
cov_options:
    cov_type: 'gaus_internal'
    fsky: 0.3
```

| Key | Type | Description |
|-----|------|-------------|
| `fsky` | `float` | Fraction of sky observed. |

#### Option 2: `SRD` — Pre-computed SRD covariance

```yaml
cov_options:
    cov_type: 'SRD'
    SRD_cov_path: "{{ env['AUGUR_DIR'] }}/data/Y1_3x2_SRD_cov.npy"
```

| Key | Type | Description |
|-----|------|-------------|
| `SRD_cov_path` | `str` | Path to `.npy` file with the covariance matrix. Y1 vs Y10 is auto-detected from the filename. |

#### Option 3: `tjpcov` — TJPCov package

```yaml
cov_options:
    cov_type: 'tjpcov'
    fsky: 0.3
    IA: 0.0
    binning_info:
        ell_edges: np.geomspace(20, 15000, 21, endpoint=True).astype(np.int32)
```

| Key | Type | Description |
|-----|------|-------------|
| `fsky` | `float` | Fraction of sky. |
| `IA` | `float` or `None` | IA amplitude for the covariance. |
| `binning_info.ell_edges` | `str` (evaluable) | Multipole bin edges for TJPCov. |

---

### `fisher`

Controls the Fisher matrix forecast.

#### Parameter specification

There are two mutually exclusive ways to specify which parameters to vary. If
both are present, `parameters` takes precedence.

**Mode A — `var_pars`** (vary around fiducial values from `cosmo`/`systematics`):

```yaml
fisher:
    var_pars: ['Omega_c', 'sigma8', 'w0', 'wa', 'lens0_bias', 'ia_bias']
```

**Mode B — `parameters`** (explicit bounds and fiducial):

```yaml
fisher:
    parameters:
        Omega_c: [0.06, 0.26642, 0.46]   # [min, fiducial, max]
        sigma8: [0.3, 0.831, 1.2]
        w0: [-3.0, -1.0, -0.33]
        ia_bias: [-3.0, 5.92, 8.0]
```

Each value is `[min, fiducial, max]`. The min/max define bounds for normalized
stepping. A scalar value can also be given as the fiducial.

#### Derivative options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `step` | `float` | `0.01` | Step size for numerical differentiation. |
| `derivative_method` | `str` | `'5pt_stencil'` | Method for numerical derivatives. Options: `'5pt_stencil'`, `'numdifftools'`, `'derivkit'`. |
| `derivative_args` | `dict` | `{}` | Extra keyword arguments passed to the derivative calculator (used with `'derivkit'`). |

#### Parameter transformations

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `transform_S8` | `bool` | `False` | Replace `sigma8` with $S_8 = \sigma_8 \sqrt{\Omega_m / 0.3}$ in the Fisher matrix. Requires `sigma8` and `Omega_c` to be varied. |
| `transform_Omega_m` | `bool` | `False` | Replace `Omega_c` with $\Omega_m = \Omega_c + \Omega_b + \Omega_\nu$. Requires `Omega_c` to be varied. |

#### Gaussian priors

```yaml
fisher:
    gaussian_priors:
        Omega_c: 0.01
        sigma8: 0.05
```

Adds $1/\sigma^2$ to the diagonal of the Fisher matrix for each specified
parameter.

#### Fisher bias

Computes parameter biases from unmodeled systematics:

$$b_i = F^{-1}_{ij} \sum_{\ell m} \Delta d_\ell \, C^{-1}_{\ell m} \frac{\partial d_m}{\partial \theta_j}$$

```yaml
fisher:
    fisher_bias:
        biased_dv: ''   # Path to file, or empty to compute internally
        bias_params:
            Omega_c: 0.27
            lens0_bias: 1.3
            lens0_delta_z: 0.01
```

| Key | Type | Description |
|-----|------|-------------|
| `biased_dv` | `str` | Path to FITS/ASCII file with column `dv_sys` containing the shifted data vector. If empty, computed internally from `bias_params`. |
| `bias_params` | `dict` | Shifted parameter values used to compute $\Delta \mathbf{d} = \mathbf{d}_{\mathrm{biased}} - \mathbf{d}_{\mathrm{fid}}$. |

#### Output paths

| Key | Type | Description |
|-----|------|-------------|
| `output` | `str` | Path for the Fisher matrix text file. Also used as base for derivative files (`.theory_vector`, `.derivatives`). |
| `fid_output` | `str` | Path for fiducial parameter values. Also base for `.biased_params`. |

---

### `postprocess`

Controls Fisher matrix visualization and figure-of-merit computation.

```yaml
postprocess:
    latex_table: "{{ env['AUGUR_DIR'] }}/output/latex_table.tex"
    triangle_plot: "{{ env['AUGUR_DIR'] }}/output/triangle_plot.pdf"
    outdir: "{{ env['AUGUR_DIR'] }}/output/"
    facecolor: blue
    pairplots: [(w0, wa), (omega_c, sigma8)]
    CL:
        - 0.68
        - 0.95
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `triangle_plot` | `str` | — | Output path for the triangle (corner) plot. |
| `latex_table` | `str` | — | Output path for LaTeX table with FoM, $\sigma_{w_0}$, $\sigma_{w_a}$. |
| `outdir` | `str` | — | Output directory for pair plots. |
| `pairplots` | `list[tuple]` | `[]` | List of parameter pairs for individual 2D contour plots. |
| `CL` | `float` or `list[float]` | `[0.68]` | Confidence level(s) for ellipses. Multiple values produce nested contours. |
| `facecolor` | `str` | `'none'` | Fill color for ellipses. |
| `linecolor` | `str` | (auto) | Edge color for ellipses. |
| `linestyle` | `str` | `'-'` | Matplotlib line style. |
| `linewidth` | `float` | `1` | Line width for ellipses. |
| `size` | `tuple` | `(48, 48)` | Figure size in inches. |
| `labels` | `list` | (auto) | Custom axis labels for the triangle plot. |
| `centers` | `list` | (fiducial) | Center values for the ellipses. |

---

## Output Files

Running a full Augur pipeline produces the following files:

| File | Stage | Contents |
|------|-------|----------|
| `{fiducial_sacc_path}` | generate | SACC FITS file with fiducial data vector + covariance |
| `{output}` | analyze | Fisher matrix (text) |
| `{output}.theory_vector` | analyze | Fiducial theory vector |
| `{output}.derivatives` | analyze | Numerical derivatives matrix |
| `{output}.priors_only` | analyze | Gaussian prior contribution |
| `{output}.with_priors` | analyze | Fisher matrix + priors combined |
| `{fid_output}` | analyze | Fiducial parameter values |
| `{fid_output}.biased_params` | analyze | Fisher bias on each parameter |
| `{output}.theory_vector_biased` | analyze | Biased $C_\ell$ difference vector |
| `{triangle_plot}` | postprocess | Triangle plot (PDF) |
| `{latex_table}` | postprocess | LaTeX FoM table |
| `{outdir}/*.pdf` | postprocess | Individual pair plots |

---

## Quick-Start Checklist

1. **Set environment variables** — e.g. `export AUGUR_DIR=/path/to/augur`.
2. **Prepare $n(z)$ files** — either use SRD analytic forms or provide text/numpy files.
3. **Choose a covariance** — `gaus_internal` for quick runs, `SRD` for pre-computed, or `tjpcov` for on-the-fly.
4. **Define systematics** — set fiducial values for all nuisance parameters you plan to vary.
5. **Configure the factory** — use `Firecrown_Factory` to specify which systematics enter the likelihood.
6. **Select Fisher parameters** — list parameters to vary in `fisher.parameters` with `[min, fid, max]`.
7. **Run** — `augur config.yml` or use the Python API.

See [config_test.yml](config_test.yml), [srd_y1_3x2.yml](srd_y1_3x2.yml),
[srd_y10_3x2.yml](srd_y10_3x2.yml), and [srd_y1_3x2_mg.yml](srd_y1_3x2_mg.yml)
for complete working examples.

---

## Example Files

| File | Description |
|------|-------------|
| [config_test.yml](config_test.yml) | Test config combining legacy-path source/lens keys with `Firecrown_Factory`. |
| [srd_y1_3x2.yml](srd_y1_3x2.yml) | SRD Year 1 3×2pt (factory path). |
| [srd_y10_3x2.yml](srd_y10_3x2.yml) | SRD Year 10 3×2pt (factory path). |
| [srd_y1_3x2_mg.yml](srd_y1_3x2_mg.yml) | SRD Year 1 3×2pt with modified gravity ($\mu$–$\Sigma$). |
| [srd_y1_3x2_like.py](srd_y1_3x2_like.py) | CosmoSIS likelihood module that wraps `augur.generate`. |
| [srd_y1_3x2_cosmosis.ini](srd_y1_3x2_cosmosis.ini) | CosmoSIS pipeline config for sampling with Augur's likelihood. |
| [srd_y1_3x2_values.ini](srd_y1_3x2_values.ini) | CosmoSIS parameter priors/values. |
| `cov_srd/` | Helper scripts for generating SRD covariance matrices with TJPCov. |
