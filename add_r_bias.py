#add r bias to already generated maps with calcspec

import argparse
import numpy as np
import h5py
import camb
from camb import model, initialpower
import phase2_common as ph2

# Handle command line arguments
parser = argparse.ArgumentParser()
# Positional arguments
parser.add_argument('field', type=int, help='can be 1 or 2')
parser.add_argument('year', type=int, help='can be 7, 10, or 20')
# Optional arguments
parser.add_argument('--rlz', type=int, nargs=2, default=[0,100],
                    help='realization start and stop indices')
parser.add_argument('--nlat', type=int, default=3,
                    help='can be 3 (default), 4, or 5')
parser.add_argument('--split', action='store_true',
                    help='set flag to use MF split bands')
parser.add_argument('--pbs', action='store_true',
                    help='set flag to use performance-based scaling for SAT')
parser.add_argument('--rbias', type=float, default=0.002, help='r bias to add')
parser.add_argument('--dry-run', action='store_true', help='print configuration and exit')

def generate_tensor_bb_spectrum(input_r, lmax):
    #generate r BB spectrum
    #set up CAMB parameters (planck)
    params = camb.CAMBparams()
    params.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.1200, mnu=0.06, omk=0, tau=0.0544)
    params.InitPower.set_params(As=2.1e-9, ns=0.9649, r=input_r)
    params.set_for_lmax(lmax, lens_potential_accuracy=0)
    params.WantTensors = True

    #power spectra calculation
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit='muK')

    #only bb power spectrum
    tensor_bb = powers['tensor'][:, 2]
    return tensor_bb

def identify_bb_spectra(spec):
    #find bb spectra and exclude all lensing templates
    bb_indices = []
    bb_info = []

    spec_names = spec.str()
    
    #look for bb spectra
    for i, name in enumerate(spec_names):
        if name.count('B') == 2:
            #exclude lensing templates
            if 'LT' not in name:
                bb_indices.append(i)
                bb_info.append((i, name))
    return bb_indices, bb_info

def get_bpwf_mapping(spec, bpwf):
    #match spectra to appropriate bpwf function
    spec_names = spec.str()
    #get all indices and names so matching up works
    all_bb_indices = []
    all_bb_names = []

    #look for all bb spectra
    for i, name in enumerate(spec_names):
        if name.count('B') == 2:
            all_bb_indices.append(i)
            all_bb_names.append(name)    
    
    bpwf_mapping = {}
    non_lt_bb_indices = []

    #determining which bpwfs go with which spectra
    bpwf_idx = 0
    for i, (spec_idx, name) in enumerate(zip(all_bb_indices, all_bb_names)):
        if bpwf_idx < bpwf.nspec:
            if 'LT' not in name:
                bpwf_mapping[spec_idx] = bpwf_idx
                non_lt_bb_indices.append(spec_idx)
            bpwf_idx += 1
    return bpwf_mapping, non_lt_bb_indices

def apply_bb_window_functions_mapped(tensor_bb, bpwf, bpwf_mapping, bb_indices):
    #apply correct bpwf functions
    bias = np.zeros((len(bb_indices), bpwf.nbin))

    #getting correct indices
    for i, bb_idx in enumerate(bb_indices):
        if bb_idx in bpwf_mapping:
            bpwf_idx = bpwf_mapping[bb_idx]
            
            #apply bpwfs
            for j in range(bpwf.nbin):
                wf_length = len(bpwf.bpwf[bpwf_idx]['BB'][j, :])
                bias[i, j] = np.sum(bpwf.bpwf[bpwf_idx]['BB'][j, :] * tensor_bb[:wf_length])    
    return bias

def add_bias_to_spectra():
    #only add to bb non-lt spectra        
    #original spectra filename
    input_file = f'spectra/phase2_spec_comb_f{args.field}_y{args.year}_n{args.nlat}_'
    if args.split:
        input_file += 'split_'
    else:
        input_file += 'full_'
    if args.pbs:
        input_file += 'withpbs_'
    else:
        input_file += 'nopbs_'
    input_file += f'{args.rlz[0]:03d}_{args.rlz[1]:03d}.h5'
        
    #load original spectra
    with h5py.File(input_file, 'r') as f:
        spec = ph2.XSpec.from_hdf5(f)
    
    #identify bb cross spectra, exclude lt
    bb_indices, bb_info = identify_bb_spectra(spec)
    
    #get bb bpwfs
    bpwf = ph2.get_bpwf(args.field, split_bands=args.split)
    
    #map bb spectra to correct bpwf
    bpwf_mapping, mapped_bb_indices = get_bpwf_mapping(spec, bpwf)
    
    #only bb spectra with bpwfs
    bb_indices = [idx for idx in bb_indices if idx in bpwf_mapping]
    bb_info = [(idx, name) for idx, name in bb_info if idx in bpwf_mapping]
        
    #generate r spectrum
    tensor_bb = generate_tensor_bb_spectrum(args.rbias, lmax=bpwf.lmax())

    #apply bpwfs to spectra
    bias = apply_bb_window_functions_mapped(tensor_bb, bpwf, bpwf_mapping, bb_indices)
    n_spectra = len(bb_indices)
    
    #add bias to only bb spectra
    for i in range(n_spectra):
        full_spec_idx = bb_indices[i]   
        for ell_idx in range(min(bpwf.nbin, spec.spec.shape[1])):
            for real_idx in range(spec.spec.shape[2]):
                spec.spec[full_spec_idx, ell_idx, real_idx] += bias[i, ell_idx]
        max_bias = np.max(np.abs(bias[i, :]))
        spec_info = bb_info[i]

    #Save biased spectra
    savefile = f'spectra/phase2_spec_comb_f{args.field}_y{args.year}_n{args.nlat}_'
    if args.split:
        savefile += 'split_'
    else:
        savefile += 'full_'
    if args.pbs:
        savefile += 'withpbs_'
    else:
        savefile += 'nopbs_'
    savefile += f'{args.rlz[0]:03d}_{args.rlz[1]:03d}_rbias{args.rbias:.0e}.h5'
    
    with h5py.File(savefile, 'w') as f:
        spec.to_hdf5(f)

if __name__ == '__main__':
    # Command line arguments
    args = parser.parse_args()
    
    # Print arguments
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
    print(f'rlz {args.rlz[0]}--{args.rlz[1]}')
    print(f'r bias = {args.rbias} (BB spectra only)')
    if args.dry_run:
        quit()
    
    add_bias_to_spectra()