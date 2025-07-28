#determine bandpass tolerance given r bias constraint

import argparse
import numpy as np
import h5py
import phase2_common as ph2
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import os
from phase2_common import Bandpass

# Handle command line arguments
parser = argparse.ArgumentParser()
# Positional arguments
parser.add_argument('field', type=int,
                    help='can be 1 or 2')
parser.add_argument('year', type=int,
                    help='can be 7, 10, or 20')
parser.add_argument('--unbiased-file', required=True,
                    help='path to file containing unbiased results') #access unbiased results
# Optional arguments (the same as usual)
parser.add_argument('--rlz', type=int, nargs=2, default=[0,100],
                    help='realization start and stop indices')
parser.add_argument('--nlat', type=int, default=3,
                    help='can be 3 (default), 4, or 5')
parser.add_argument('--split', action='store_true',
                    help='set flag to use MF split bands')
parser.add_argument('--pbs', action='store_true',
                    help='set flag to use performance-based scaling for SAT')
parser.add_argument('--noffdiag', type=int, default=0,
                    help='number of off-diagonal blocks to keep in BPCM')
parser.add_argument('--r-bias-tol', type=float, default=0.0001, 
                    help='maximum allowed bias on r')
parser.add_argument('--n-realizations', type=int, default=10,
                    help='number of realizations to use for testing')
#single band pct handling
parser.add_argument('--single-band', type=str, default=None,
                    help='search for percent tolerance on a single band (e.g., LF-1, MF-2)')
parser.add_argument('--max-bias-search', type=float, default=5.0,
                    help='maximum bandpass bias to search in percent')
#all band ghz handling
parser.add_argument('--all-bands-ghz', action='store_true',
                    help='search for GHz tolerance applied to all bands')
parser.add_argument('--max-ghz-search', type=float, default=10.0,
                    help='maximum GHz bias to search when using --all-bands-ghz')

def starting_guess(field): #ripped from mlsearch
    if field == 1:
        guess = {'r': 0.0, 'Alens': 1.0, 
                 'A_d': 13.7, 'alpha_d': -0.68, 'beta_d': 1.64, 'T_d': 19.6, 'EEBB_d': 2.0,
                 'A_s': 1.1, 'alpha_s': -1.1, 'beta_s': -3.1, 'EEBB_s': 2.0,
                 'epsilon': 0.025, 'Delta_d': 0.999, 'gamma_d': 0.0,
                 'Delta_s': 0.999, 'gamma_s': 0.0}
    elif field == 2:
        guess = {'r': 0.0, 'Alens': 1.0,
                 'A_d': 59.5, 'alpha_d': -0.55, 'beta_d': 1.54, 'T_d': 19.6, 'EEBB_d': 2.0,
                 'A_s': 1.1, 'alpha_s': -1.5, 'beta_s': -3.0, 'EEBB_s': 2.0,
                 'epsilon': 0.09, 'Delta_d': 0.997, 'gamma_d': 0.0,
                 'Delta_s': 0.998, 'gamma_s': 0.0}
    else:
        raise ValueError('field must be 1-2')
    return guess

def load_unbiased_results(filename): #load existing unbiased results
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Unbiased results not found: {filename}")
    #load results
    results = np.load(filename)
    #get rid of failed searches
    mask = results[0,:] == 0
    r_values = results[2, mask]
    print(f"Loaded {np.sum(mask)} realizations from {filename}")
    return r_values, mask

def apply_ghz_bias_to_all_bands(bands_dict, ghz_bias): #apply ghz bias to all bands
    if ghz_bias == 0.0:
        return bands_dict
    
    biased_bands = bands_dict.copy()
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
    
    for band_name in biased_bands:
        if band_name != 'LT' and band_name in original_bandpasses:
            nu_min, nu_max = original_bandpasses[band_name]
            new_nu_min = nu_min + ghz_bias
            new_nu_max = nu_max + ghz_bias
            biased_bands[band_name]['bandpass'] = Bandpass.tophat(new_nu_min, new_nu_max)
    
    print(f"Applied {ghz_bias:+.1f} GHz bias to all bands:")
    for band_name in sorted(original_bandpasses.keys()):
        if band_name in biased_bands:
            nu_min, nu_max = original_bandpasses[band_name]
            print(f"  {band_name}: ({nu_min:.1f}, {nu_max:.1f}) → ({nu_min + ghz_bias:.1f}, {nu_max + ghz_bias:.1f}) GHz")
    return biased_bands

def run_ml_search_with_bias(data, field, year, nlat, rlz0, rlz1, split, pbs, 
                           noffdiag, band_name, bias_percent, realization_indices,
                           all_bands_ghz=None): #modified slightly from mlsearch for specified bandpass bias on one band
    original_bands = ph2.bands.copy()
    
    if all_bands_ghz is not None:
        ph2.bands = apply_ghz_bias_to_all_bands(ph2.bands, all_bands_ghz)
    elif band_name is not None:
        ph2.bands = ph2.apply_band_bias(ph2.bands, band_name, bias_percent)
    
    lik = ph2.get_likelihood(field, year, nlat, rlz0, rlz1, split, pbs)
    
    guess = starting_guess(field)
    lik.compute_fiducial_bpcm(lik.expv(guess, include_bias=False),
                              noffdiag=noffdiag, mask_noise=True)
    
    free = ['r', 'A_d','alpha_d', 'beta_d', 'A_s', 'alpha_s', 'beta_s', 'epsilon',
            'Delta_d', 'Delta_s'] 
    limits = {'beta_d': [1.0, 2.0], 'alpha_d': [-2.0, 0.5],
              'beta_s': [-4.0, -2.0], 'alpha_s': [-2.0, 0.5],
              'epsilon': [-1,1], 'Delta_d': [0.5,1.1], 'Delta_s': [0.5,1.1]}
    method = 'L-BFGS-B'
    options = {'maxls': 30}
    
    #dont run for all 100 realizations, only do the ones specified in argument above
    r_values = []
    
    for i in realization_indices:
        (result, fval, status) = lik.mlsearch(data[:,:,i], guess, free=free, 
                                              limits=limits, method=method, options=options)
        if status == 0:  #once again get rid of any failed searches
            r_values.append(result['r'])
    ph2.bands = original_bands
    return np.array(r_values)

def find_ghz_tolerance_all_bands(data, field, year, nlat, rlz0, rlz1, split, pbs, 
                                noffdiag, r_bias_tol, r_baseline, 
                                realization_indices, max_ghz_search=3.0):
    #find max pos and neg bias to keep r under tolerance    
    print(f"\nFinding GHz tolerance for all bands...")
    
    r_baseline_mean = np.mean(r_baseline)
    print(f"Baseline r mean from existing data: {r_baseline_mean:.6f}")
    
    #evaluate r bias for given ghz shift
    def evaluate_r_bias(ghz_bias):
        r_biased = run_ml_search_with_bias(data, field, year, nlat, rlz0, rlz1, 
                                          split, pbs, noffdiag, None, 0.0,
                                          realization_indices, all_bands_ghz=ghz_bias)
        if len(r_biased) == 0:
            return np.inf  # if all fits failed
        r_bias = np.mean(r_biased) - r_baseline_mean
        return r_bias
    
    #find positive GHz tolerance (binary search)
    print("Searching for positive GHz tolerance...")
    pos_low, pos_high = 0.0, max_ghz_search
    while pos_high - pos_low > 0.01:  # 0.01 GHz precision
        mid = (pos_low + pos_high) / 2
        r_bias = evaluate_r_bias(mid)
        print(f"  Testing +{mid:.2f} GHz: r_bias = {r_bias:.6f}")
        if abs(r_bias) > r_bias_tol:
            pos_high = mid
        else:
            pos_low = mid
    pos_tolerance = pos_low
    
    #find negative GHz tolerance (binary search)
    print("Searching for negative GHz tolerance...")
    neg_low, neg_high = -max_ghz_search, 0.0
    while neg_high - neg_low > 0.01:  # 0.01 GHz precision
        mid = (neg_low + neg_high) / 2
        r_bias = evaluate_r_bias(mid)
        print(f"  Testing {mid:.2f} GHz: r_bias = {r_bias:.6f}")
        if abs(r_bias) > r_bias_tol:
            neg_low = mid
        else:
            neg_high = mid
    neg_tolerance = neg_high
    
    return neg_tolerance, pos_tolerance

def find_bias_tolerance(data, field, year, nlat, rlz0, rlz1, split, pbs, 
                       noffdiag, band_name, r_bias_tol, r_baseline, 
                       realization_indices, max_search=5.0):
    #find max pos and neg bias to keep r under tolerance    
    print(f"\nFinding bias tolerance for {band_name}...")

    r_baseline_mean = np.mean(r_baseline)
    print(f"Baseline r mean from existing data: {r_baseline_mean:.6f}")
    
    #evaluate r bias for given bandpass bias
    def evaluate_r_bias(bp_bias):
        r_biased = run_ml_search_with_bias(data, field, year, nlat, rlz0, rlz1, 
                                          split, pbs, noffdiag, band_name, bp_bias,
                                          realization_indices)
        if len(r_biased) == 0:
            return np.inf  #if all fits failed
        r_bias = np.mean(r_biased) - r_baseline_mean
        return r_bias
    
    #find positive bandpaass bias tolerance (binary search)
    print("Searching for positive bias tolerance...")
    pos_low, pos_high = 0.0, max_search
    while pos_high - pos_low > 0.1:  #0.1% precision
        mid = (pos_low + pos_high) / 2
        r_bias = evaluate_r_bias(mid)
        print(f"  Testing {mid:.1f}%: r_bias = {r_bias:.6f}")
        if abs(r_bias) > r_bias_tol:
            pos_high = mid
        else:
            pos_low = mid
    pos_tolerance = pos_low
    #find negative bandpaass bias tolerance (binary search)
    print("Searching for negative bias tolerance...")
    neg_low, neg_high = -max_search, 0.0
    while neg_high - neg_low > 0.1:  #0.1% precision
        mid = (neg_low + neg_high) / 2
        r_bias = evaluate_r_bias(mid)
        print(f"  Testing {mid:.1f}%: r_bias = {r_bias:.6f}")
        if abs(r_bias) > r_bias_tol:
            neg_low = mid
        else:
            neg_high = mid
    neg_tolerance = neg_high
    return neg_tolerance, pos_tolerance

if __name__ == '__main__':
    # Command line arguments
    args = parser.parse_args()
    
    print(f'field = {args.field}')
    print(f'year = {args.year}')
    print(f'nlat = {args.nlat}')
    if args.split:
        print('using split bands')
    else:
        print('using full bands')
    if args.pbs:
        print('using performance-based scaling')
    else:
        print('no performance-based scaling')
    print(f'r bias tolerance = {args.r_bias_tol}')
    if args.all_bands_ghz:
        print(f'searching for GHz tolerance on all bands')
        print(f'max GHz search = ±{args.max_ghz_search} GHz')
    else:
        print(f'band = {args.single_band}')
        print(f'max bias search = {args.max_bias_search}%')
    print(f'using {args.n_realizations} realizations for testing')
    print(f'loading unbiased results from: {args.unbiased_file}')
        
    #load existing unbiased results
    print("\nLoading existing unbiased ML search results...")
    r_unbiased_all, success_mask = load_unbiased_results(args.unbiased_file)
    
    #select subset of realizations to use for tolerance search
    #use the first n_realizations that were successful in the unbiased case
    successful_indices = np.where(success_mask)[0]
    n_to_use = min(args.n_realizations, len(successful_indices))
    realization_indices = successful_indices[:n_to_use]
    r_unbiased_subset = r_unbiased_all[:n_to_use]
    print(f"Using {n_to_use} realizations for tolerance search")
    print(f"Realization indices: {realization_indices}")
    
    #read CMB+fg+noise spectra
    print("\nReading data...")
    data = ph2.get_spectra('comb', args.field, args.year, args.nlat, 
                          args.rlz[0], args.rlz[1], split_bands=args.split, 
                          pbscaling=args.pbs)
    
    if args.all_bands_ghz:
        neg_tol, pos_tol = find_ghz_tolerance_all_bands(data, args.field, args.year, 
                                                        args.nlat, args.rlz[0], args.rlz[1], 
                                                        args.split, args.pbs, args.noffdiag, 
                                                        args.r_bias_tol, r_unbiased_subset, 
                                                        realization_indices, args.max_ghz_search)
        
        print(f"\nFinal result for all bands: [{neg_tol:.2f} GHz, +{pos_tol:.2f} GHz]")
        
        #save results
        savefile = f'bandpass_tolerance/ph2_tolerance_f{args.field}_y{args.year}_n{args.nlat}_diag{args.noffdiag}'
        if args.split:
            savefile += '_split'
        else:
            savefile += '_full'
        if args.pbs:
            savefile += '_withpbs'
        else:
            savefile += '_nopbs'
        savefile += f'_all-ghz_rtol{args.r_bias_tol:.1e}_nrlz{n_to_use}.npy'
        
        save_array = np.array([neg_tol, pos_tol])
        np.save(savefile, save_array)
        print(f"Results saved to {savefile}")
        
    else:
        #find tolerance for the specified band
        neg_tol, pos_tol = find_bias_tolerance(data, args.field, args.year, 
                                              args.nlat, args.rlz[0], args.rlz[1], 
                                              args.split, args.pbs, args.noffdiag, 
                                              args.single_band, args.r_bias_tol, 
                                              r_unbiased_subset, realization_indices,
                                              args.max_bias_search)
        
        print(f"\nFinal result for {args.single_band}: [{neg_tol:.2f}%, +{pos_tol:.2f}%]")
        
        #save results
        savefile = f'bandpass_tolerance/ph2_tolerance_f{args.field}_y{args.year}_n{args.nlat}_diag{args.noffdiag}'
        if args.split:
            savefile += '_split'
        else:
            savefile += '_full'
        if args.pbs:
            savefile += '_withpbs'
        else:
            savefile += '_nopbs'
        savefile += f'_{args.single_band}_rtol{args.r_bias_tol:.1e}_nrlz{n_to_use}.npy'
        
        band_index = valid_bands.index(args.single_band)
        save_array = np.array([band_index, neg_tol, pos_tol])
        np.save(savefile, save_array)
        print(f"Results saved to {savefile}")