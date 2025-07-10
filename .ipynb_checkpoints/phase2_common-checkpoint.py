"""
================
phase2_common.sh
================

Definitions and file paths for Chile optimization phase 2 analysis.

Requires the following symlinks:
  - phase2 -> /global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/
  - spectra -> wherever you want to save the calculated spectra
  - mlsearch -> wherever you want to save the maximum likelihood search results
  - lowellbb -> /global/cfs/cdirs/cmbs4/awg/lowellbb

"""

import numpy as np
import healpy as hp
from scipy.stats import gmean
import h5py
import pymaster as nmt
from s4bb.bandpass import Bandpass
from s4bb.bpcov import BpCov_signoi
from s4bb.likelihood import Likelihood
from s4bb.models import Model_cmb, Model_fg
from s4bb.spectra import XSpec, CalcSpec_namaster
from s4bb.util import MapDef, mapind, specind

# Epochs that were simulated for phase 2
VALID_YEARS = [7,10,20]

# NSIDE=512 for phase 2 sims
NSIDE = 512

# CMB-S4 SAT bands
bands = {'LF-1': {'freq': 26,
		  'bandpass': Bandpass.tophat(21.5, 28.0), 
		  'fwhm_arcmin': 79.2, 'lensing_template': False,
                  'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f030_30years_cov.fits'},
	 'LF-2': {'freq': 39,
		  'bandpass': Bandpass.tophat(28.0, 45.0),
		  'fwhm_arcmin': 56.6, 'lensing_template': False,
                  'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f040_30years_cov.fits'},
	 'MF1-1': {'freq': 85,
		   'bandpass': Bandpass.tophat(74.8, 95.2),
		   'fwhm_arcmin': 22.9, 'lensing_template': False,
                   'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f085_90years_cov.fits'},
	 'MF2-1': {'freq': 95,
		   'bandpass': Bandpass.tophat(83.6, 106.4),
		   'fwhm_arcmin': 20.6, 'lensing_template': False,
                   'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f095_90years_cov.fits'},
         'MF-1': {'freq': 90,
                  'bandpass': Bandpass.tophat(77.0 , 106.0),
                  'fwhm_arcmin': 21.4, 'lensing_template': False,
                  'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f090_180years_cov.fits'},
	 'MF1-2': {'freq': 145,
		   'bandpass': Bandpass.tophat(129.1, 161.0),
		   'fwhm_arcmin': 14.2, 'lensing_template': False,
                   'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f145_90years_cov.fits'},
	 'MF2-2': {'freq': 155,
		   'bandpass': Bandpass.tophat(138.0, 172.1),
		   'fwhm_arcmin': 13.5, 'lensing_template': False,
                   'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f155_90years_cov.fits'},
         'MF-2': {'freq': 150,
                  'bandpass': Bandpass.tophat(128.0, 169.0),
                  'fwhm_arcmin': 14.0, 'lensing_template': False,
                  'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f150_180years_cov.fits'},
	 'HF-1': {'freq': 227, 
		  'bandpass': Bandpass.tophat(198.0, 256.0),
		  'fwhm_arcmin': 9.4, 'lensing_template': False,
                  'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f220_60years_cov.fits'},
	 'HF-2': {'freq': 286,
		  'bandpass': Bandpass.tophat(256.0, 315.0), #MODIFYING THESE BANDS TO SEE WHAT HAPPENS (
		  'fwhm_arcmin': 7.8, 'lensing_template': False,
                  'cov': '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/noise_depth/sun90max_f280_60years_cov.fits'},
	 'LT': {'freq': None, 'bandpass': None, 'fwhm_arcmin': None,
		'lensing_template': True}
	 }

# Bands to use for "with split bands" vs "no split bands" cases.
with_split_bands = ['LF-1', 'LF-2', 'MF1-1', 'MF2-1', 'MF1-2', 'MF2-2', 'HF-1', 'HF-2', 'LT']
no_split_bands = ['LF-1', 'LF-2', 'MF-1', 'MF-2', 'HF-1', 'HF-2', 'LT']

#ADDED FOR BIASING BANDS
def apply_band_bias(bands_dict, bias_band = None, bias_percent = 0.0):
    if bias_band is None or bias_percent == 0.0:
        return bands_dict
    
    biased_bands = bands_dict.copy()    
    bias_factor = 1.0 + (bias_percent / 100.0)
    band_info = biased_bands[bias_band]
    
    original_bandpasses = {
        'LF-1': (21.5, 28.0),
        'LF-2': (28.0, 45.0),
        'MF1-1': (74.8, 95.2),
        'MF2-1': (83.6, 106.4),
        'MF-1': (77.0, 106.0),
        'MF1-2': (129.1, 161.0),
        'MF2-2': (138.0, 172.1),
        'MF-2': (128.0, 169.0),
        'HF-1': (198.0, 256.0),
        'HF-2': (256.0, 315.0)
    }
    
    nu_min, nu_max = original_bandpasses[bias_band]
    new_nu_min = nu_min * bias_factor
    new_nu_max = nu_max * bias_factor
    biased_bands[bias_band]['bandpass'] = Bandpass.tophat(new_nu_min, new_nu_max)
        
    print(f"Applied {bias_percent:+.1f}% bias to {bias_band}:")
    print(f"  Original: ({nu_min:.1f}, {nu_max:.1f}) GHz")
    print(f"  Biased:   ({new_nu_min:.1f}, {new_nu_max:.1f}) GHz")    
    return biased_bands


# Ell bins: delta-ell=20, starting from ell=30
bin_low = np.arange(30, 500, 20)
bin_high = bin_low + 20
bins = np.stack((bin_low, bin_high))

# Central pixels for the two subfields
pixcen = { 1: 2636910, 2: 1687443 }

# Map apodization used for this analysis
def get_apod(field, band):
    """Make apodization map for specified field"""

    # Get angular offset between each pixel and the field center
    (x, y, z) = hp.pix2vec(NSIDE, range(hp.nside2npix(NSIDE)))
    (x0, y0, z0) = hp.pix2vec(NSIDE, pixcen[field])
    ang = np.arccos(x * x0 + y * y0 + z * z0)

    # Several different options to try here:
    
    # 1. Gaussian apodization with sigma = 10 degrees
    if band == 'gauss':
        apod = np.exp(-0.5 * ang**2 / np.radians(10)**2)
    # 2. Geometric mean across frequency bands of Q/U inverse variance
    elif band == 'gmean':
        band_list = ['LF-1', 'LF-2', 'MF1-1', 'MF2-1', 'MF1-2', 'MF2-2', 'HF-1', 'HF-2']
        ivar = np.zeros(shape=(len(band_list), hp.nside2npix(NSIDE)))
        for (i,val) in enumerate(band_list):
            cov = hp.read_map(bands[val]['cov'], field=(0,1,2))
            with np.errstate(divide='ignore'):
                ivar[i,:] = 1 / np.sqrt(cov[1,:] * cov[2,:])
        ivar[np.isinf(ivar)] = 0.0
        apod = gmean(ivar, axis=0)
    else:
        # 3. Use Q/U inverse variance for the specified frequency band
        #    For lensing template, use the MF-1 apodization (arbitrary choice)
        if band == 'LT':
            return get_apod(field, 'MF-1')    
        # Read TQU covariance
        apod = hp.read_map(bands[band]['cov'], field=(0,1,2))
        # Inverse of geometric mean of Q and U variances
        with np.errstate(divide='ignore'):
            apod = 1 / np.sqrt(apod[1,:] * apod[2,:])
            apod[np.isinf(apod)] = 0.0

    # For any of the above options, we now select the subfield
    # Zero out apodization for any pixels more than 45 degrees from subfield center
    apod[ang > np.pi / 4] = 0.0
    # Smooth edges
    apod = nmt.mask_apodization(apod, 5.0, apotype='C1')
    return apod / np.max(apod)

# get_maplist provides definitions for Phase-1 QU sim maps
def get_maplist(simtype, split_bands=True):
    """Construct list of MapDef objects corresponding to input sim maps"""

    # Do we use split bands?
    if split_bands:
        band_list = with_split_bands
    else:
        band_list = no_split_bands
        
    if simtype.startswith('signoi'):
        # Recursively call this function for LLCDM and noise simtypes.
        maplist_sig = get_maplist('llcdm', split_bands=split_bands)
        maplist_noi = get_maplist('noise', split_bands=split_bands)
        maplist = maplist_sig + maplist_noi
    else:
        # Check whether this is a signal-only or noise-only sim
        if simtype.startswith('llcdm') or simtype.startswith('tensor'):
            stype = 'signal'
        elif simtype.startswith('noise'):
            stype = 'noise'
        else:
            stype = None
        maplist = []
        for band in band_list:
            maplist.append(MapDef(band, 'QU',
                                  bandpass=bands[band]['bandpass'],
                                  fwhm_arcmin=bands[band]['fwhm_arcmin'],
                                  lensing_template=bands[band]['lensing_template'],
                                  simtype=stype))
    return maplist

def read_noise_map(band, yr, nlat, rlz, pbscaling=False):
    """Read noise map, QU only"""

    # Directory for noise sims
    if yr not in VALID_YEARS:
        raise ValueError('invalid number of years')
    # Read noise map
    if band == 'LT':
        # Basically negligible white noise level (but non-zero to
        # avoid problems)
        LT_NOISE = 1e-2 # rms noise, in uK
        m = LT_NOISE * np.random.randn(2, hp.nside2npix(NSIDE))
    else:
        nu = bands[band]['freq']
        if pbscaling:
            dir = f'/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/with_pbscaling/noise_{yr:02d}_years/'
            mapname = f'phase2_noise_f{nu:03d}_SAT_mc_{rlz:04d}.fits'
        else:
            dir = f'/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/no_pbscaling/noise_{yr:02d}_years/'
            mapname = f'phase2_noise_f{nu:03d}_SAT90_mc_{rlz:04d}.fits'            
        m = hp.read_map(dir + mapname, field=(1,2))
        m[m == hp.UNSEEN] = 0.0
        # Convert from K to uK
        m = m * 1e6        
    return m

def read_llcdm_map(band, yr, nlat, rlz):
    """Read CMB map, QU only"""

    # Directory for combined maps
    if yr not in VALID_YEARS:
        raise ValueError('invalid number of years')
    # Read map
    if band == 'LT':
        if yr == 20:
            dir = f'/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/lensing_templates/no_pbscaling_no_artifact/v0/years{yr:02d}_lats{nlat:1d}/'
        else:
            dir = f'/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/lensing_templates/no_pbscaling/v0/years{yr:02d}_lats{nlat:1d}/'
        almname = f'blt_p012_e013_lmax1024_yr{yr:02d}_lat{nlat:1d}_sim{rlz:03d}.fits'
        Blm = hp.read_alm(dir + almname, hdu=1)
        # These files include Blm only, set Tlm and Elm to zero
        alm = np.zeros(shape=(3,len(Blm)), dtype=Blm.dtype)
        alm[2,:] = Blm
        # Lensing templates are already in uK
        # Convert back to a map, keep Q/U only
        m = hp.alm2map(alm, NSIDE, pol=True)[1:]
    else:
        dir = '/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/cmb/sat/'        
        nu = bands[band]['freq']
        mapname = f'phase2_cmb_f{nu:03d}_mc_{rlz:04d}.fits'
        m = hp.read_map(dir + mapname, field=(1,2))
        m[m == hp.UNSEEN] = 0.0
        # Convert from K to uK
        m = m * 1e6
    return m

def read_comb_map(band, yr, nlat, rlz, pbscaling=False):
    """Read CMB+fg+noise combined map, QU only"""

    if yr not in VALID_YEARS:
        raise ValueError('invalid number of years')    
    if band == 'LT':
        return read_llcdm_map(band, yr, nlat, rlz)

    # Read map
    nu = bands[band]['freq']
    if pbscaling:
        dir = f'/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/with_pbscaling/total_{yr:02d}_years/'
        mapname = f'phase2_total_f{nu:03d}_SAT_mc_{rlz:04d}.fits'
    else:
        dir = f'/global/cfs/cdirs/cmbs4/chile_optimization/simulations/phase2/no_pbscaling/total_{yr:02d}_years/'
        mapname = f'phase2_total_f{nu:03d}_SAT90_mc_{rlz:04d}.fits'
    m = hp.read_map(dir + mapname, field=(1,2))
    m[m == hp.UNSEEN] = 0.0
    # Convert from K to uK
    m = m * 1e6
    return m

def read_maps(maplist, simtype, yr, nlat, rlz, pbscaling=False):
    """Read in maps for all bands, QU only"""

    # Read frequency maps
    maps = []
    for ml in maplist:
        if simtype == 'noise':
            m = read_noise_map(ml.name, yr, nlat, rlz, pbscaling=pbscaling)
        elif simtype == 'llcdm':
            m = read_llcdm_map(ml.name, yr, nlat, rlz)
        elif simtype == 'comb':
            m = read_comb_map(ml.name, yr, nlat, rlz, pbscaling=pbscaling)
        elif simtype == 'signoi':
            if ml.simtype == 'signal':
                m = read_llcdm_map(ml.name, yr, nlat, rlz)
            elif ml.simtype == 'noise':
                m = read_noise_map(ml.name, yr, nlat, rlz, pbscaling=pbscaling)
            else:
                raise ValueError('invalid sim type')
        else:
            raise ValueError('invalid sim type')
        maps.append(m)
    return maps

def trim_maps(apod, maps, threshold=1e-6):
    """Zero out maps in regions where apodization is below threshold"""

    # Should be one apodization for each map
    assert len(apod) == len(maps)
    # Trim maps
    for i in range(len(apod)):
        for j in range(maps[i].shape[0]):
            maps[i][j,apod[i] < threshold] = 0.0

def spectra_file(simtype, field, yr, nlat, rlz0, rlz1, split_bands=True, pbscaling=False):
    """Returns HDF5 file name for spectra"""
    #filename = '/global/cfs/cdirs/cmbs4/chile_optimization/analysis/cbischoff/phase2/spectra/phase2_spec_' USE THIS LINE WHEN FINDING COLINS SPECTRA
    filename = 'spectra/phase2_spec_'
    filename += f'{simtype}_f{field:1d}_y{yr:1d}_n{nlat:1d}_'
    if split_bands:
        filename += 'split_'
    else:
        filename += 'full_'
    if pbscaling:
        filename += 'withpbs_'
    else:
        filename += 'nopbs_'
    filename += f'{rlz0:03d}_{rlz1:03d}.h5'
    return filename

def get_spectra(simtype, field, yr, nlat, rlz0, rlz1, split_bands=True, pbscaling=False, rbias = None):
    """Loads spectra from HDF5 file"""

    filename = spectra_file(simtype, field, yr, nlat, rlz0, rlz1, split_bands=split_bands, pbscaling=pbscaling)
    #ADDED BIASED SPECTRUM CONSIDERATION
    if rbias is not None:
        filename = filename.replace('.h5', f'_rbias{rbias:.1e}.h5')
    with h5py.File(filename, 'r') as f:
        spec = XSpec.from_hdf5(f)
    # Keep BB only, ell bins 0-12
    spec = spec.select(maplist=spec.maplist[1::2], ellind=range(13))
    # Done
    return spec

def get_bpcm(field, yr, nlat, rlz0, rlz1, split_bands=True, pbscaling=False):
    """Bandpower covariance matrix from signal and noise sims"""

    # Read spectra
    spec = get_spectra('signoi', field, yr, nlat, rlz0, rlz1, split_bands=split_bands, pbscaling=pbscaling)
    # Get maplist for combined (signal+noise) sims
    comb = get_spectra('comb', field, yr, nlat, rlz0, rlz1, split_bands=split_bands, pbscaling=pbscaling)
    # Build covariance matrix
    bpcm = BpCov_signoi.from_xspec(comb.maplist, spec)
    # Done
    return bpcm

def get_bpwf(field, split_bands=True):
    """Bandpower window functions from NaMaster"""

    # Read apodization mask -- hardcoded gmean option for now
    apod = get_apod(field, 'gmean')
    # Input maps to power spectrum estimator
    maplist = get_maplist('llcdm', split_bands=split_bands)
    for m in maplist:
        m.simtype = None
    # Use pure-B estimator for all maps *except* lensing template
    pureb = [False if m.name == 'LT' else True for m in maplist]
    
    # Instantiate power spectrum estimator
    cs = CalcSpec_namaster(maplist, apod, NSIDE, bins,
                           use_Dl=True, pure_B=pureb, Bl_min=1e-12)
    # Get bandpower window functions
    wf = cs.get_bpwf(input_Dl=True)
    # Select BB spectra only, ell bins 0-12
    wf = wf.select(maplist=wf.maplist[1::2], ellind=range(13))

    # The phase 2 SAT maps roll off signal between 27 < ell < 33 with a sin profile
    highpass = np.zeros(34)
    highpass[27:34] = np.sin(np.linspace(0, np.pi/2, 7))
    for i in range(wf.nspec):
        (m0, m1) = mapind(wf.nspec, i)
        for key in wf.bpwf[i].keys():
            wf.bpwf[i][key][:,0:34] = wf.bpwf[i][key][:,0:34] * highpass[0:34]
    # Done
    return wf

def get_cmb_model(wf):
    """Model for calculating CMB expectation values"""

    # Unlensed LCDM
    Dl_unlens = np.zeros(shape=(4,wf.lmax()+1))
    Dl = np.genfromtxt('/global/cfs/cdirs/cmbs4/awg/lowellbb/sky_yy/cmb/cls/ffp10_scalCls.dat')
    ell0 = int(Dl[0,0])
    Dl_unlens[0,ell0:] = Dl[0:wf.lmax()+1-ell0,1] # TT
    Dl_unlens[1,ell0:] = Dl[0:wf.lmax()+1-ell0,2] # EE
    Dl_unlens[3,ell0:] = Dl[0:wf.lmax()+1-ell0,3] # TE
    # Lensed LCDM
    Dl_lens = np.zeros(shape=(4,wf.lmax()+1))
    Dl = np.genfromtxt('/global/cfs/cdirs/cmbs4/awg/lowellbb/sky_yy/cmb/cls/ffp10_lensedCls.dat')
    ell0 = int(Dl[0,0])
    Dl_lens[0,ell0:] = Dl[0:wf.lmax()+1-ell0,1] # TT
    Dl_lens[1,ell0:] = Dl[0:wf.lmax()+1-ell0,2] # EE
    Dl_lens[2,ell0:] = Dl[0:wf.lmax()+1-ell0,3] # BB
    Dl_lens[3,ell0:] = Dl[0:wf.lmax()+1-ell0,4] # TE
    # Tensors
    Dl_tensor = np.zeros(shape=(4,wf.lmax()+1))
    Dl = np.genfromtxt('/global/cfs/cdirs/cmbs4/awg/lowellbb/sky_yy/cmb/cls/ffp10_wtensors_tensCls.dat')
    ell0 = int(Dl[0,0])
    Dl_tensor[0,ell0:] = Dl[0:wf.lmax()+1-ell0,1] # TT
    Dl_tensor[1,ell0:] = Dl[0:wf.lmax()+1-ell0,2] # EE
    Dl_tensor[2,ell0:] = Dl[0:wf.lmax()+1-ell0,3] # BB
    Dl_tensor[3,ell0:] = Dl[0:wf.lmax()+1-ell0,4] # TE
    rval = 0.01 # read this from .ini file
    # Create Model_cmb object
    mod = Model_cmb(wf.maplist, wf, Dl_unlens, Dl_lens, Dl_tensor, rval)
    return mod

def adjust_bpwf(wf, field, yr, nlat, rlz0, rlz1, split_bands=True, pbscaling=False):
    """
    Renormalize window functions for lensing template so that expectation
    value matches mean of sims.
    
    """

    # Read LLCDM-only sims
    spec = get_spectra('llcdm', field, yr, nlat, rlz0, rlz1, split_bands=split_bands, pbscaling=pbscaling)
    # CMB model with r=0, Alens=1
    mod = get_cmb_model(wf)
    param = {'r': 0.0, 'Alens': 1.0}
    expv = mod.expv(param)
    # Loop over lensing template cross-spectra and auto-spectrum
    nmap = spec.nmap()
    for (i,m) in enumerate(spec.maplist):
        if m.name == 'LT':
            ltind = i
            break
    for i in range(nmap):
        idx = specind(nmap, i, ltind)
        wf.adjust_windowfn('BB', idx, spec[idx,:,:].mean(axis=1) / expv[idx,:])

def get_bias(field, yr, nlat, rlz0, rlz1, split_bands=True, pbscaling=False):
    """Get noise bias for specified year, field, and number of LATs"""

    # Read noise-only sims
    spec = get_spectra('noise', field, yr, nlat, rlz0, rlz1, split_bands=split_bands, pbscaling=pbscaling)
    # Average over realizations and store as XSpec object
    bias = XSpec(spec.maplist, spec.bins, spec[:].mean(axis=2))
    # Reset MapDef.simtype because it causes problems if this is set to 'noise'
    for m in bias.maplist:
        m.simtype = None
    # Done
    return bias

def get_likelihood(field, yr, nlat, rlz0, rlz1, split_bands=True, pbscaling=False):
    """Get likelihood object"""
    bias = get_bias(field, yr, nlat, rlz0, rlz1, split_bands=split_bands, pbscaling=pbscaling)
    bpcm = get_bpcm(field, yr, nlat, rlz0, rlz1, split_bands=split_bands, pbscaling=pbscaling)
    wf = get_bpwf(field, split_bands=split_bands)
    adjust_bpwf(wf, field, yr, nlat, rlz0, rlz1, split_bands=split_bands, pbscaling=pbscaling)
    mod0 = get_cmb_model(wf)
    mod1 = Model_fg(wf.maplist, wf)
    return Likelihood(bias.maplist, bias=bias, bpcm=bpcm, models=[mod0, mod1])
