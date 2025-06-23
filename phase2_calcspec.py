# phase2_calcspec.py
# ------------------
# Calculate power spectra for Chile optimization phase 2 sims

import argparse
import numpy as np
import healpy as hp
import pymaster as nmt
import git
import h5py
import phase2_common as ph2
from s4bb.spectra import CalcSpec_namaster

# Handle command line arguments
parser = argparse.ArgumentParser()
# Positional arguments
parser.add_argument('simtype', type=str,
                    help='can be noise, llcdm, comb, or signoi')
parser.add_argument('field', type=int,
                    help='can be 1 or 2')
parser.add_argument('year', type=int,
                    help='can be 7, 10, or 20')
# Optional arguments
parser.add_argument('--rlz', type=int, nargs=2, default=[0,100],
                    help='realization start and stop indices')
parser.add_argument('--nlat', type=int, default=3,
                    help='can be 3 (default), 4, or 5')
parser.add_argument('--split', action='store_true',
                    help='set flag to use MF split bands')
parser.add_argument('--pbs', action='store_true',
                    help='set flag to use performance-based scaling for SAT')
parser.add_argument('--dry-run', action='store_true',
                    help='print configuration and exit')

if __name__ == '__main__':
    # Command line arguments
    args = parser.parse_args()

    # Print arguments
    print(f'simtype = {args.simtype}')
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
    if args.dry_run:
        quit()
    
    # Code provenance
    #s4bbrepo = git.Repo('s4bb/')
    #print('s4bb version: {}'.format(s4bbrepo.head.object.hexsha))
    print('s4bb version: local files only (no git)')
    print('namaster version: {}'.format(nmt.__version__))
    
    # Input maps to power spectrum estimator
    maplist = ph2.get_maplist(args.simtype, split_bands=args.split)
    for m in maplist:
        print(m)
    # After some testing, it seems like the best apodization is to use
    # the geometric mean over frequencies of Q/U inverse variance.
    apod = [ph2.get_apod(args.field, m.name) for m in maplist]
    # Use pure-B estimator for all maps *except* lensing template
    pureb = [False if m.name == 'LT' else True for m in maplist]
    # Instantiate power spectrum estimator
    cs = CalcSpec_namaster(maplist, apod, ph2.NSIDE, ph2.bins,
                           use_Dl=True, pure_B=pureb, Bl_min=1e-12)
    
    # Calculate power spectra for all realizations
    print(args.rlz[0])
    maps = ph2.read_maps(maplist, args.simtype, args.year, args.nlat,
                         args.rlz[0], pbscaling=args.pbs)
    ph2.trim_maps(apod, maps)
    spec = cs.calc(maps)
    for i in range(args.rlz[0]+1, args.rlz[1]):
        print(i)
        maps = ph2.read_maps(maplist, args.simtype, args.year, args.nlat,
                             i, pbscaling=args.pbs)
        ph2.trim_maps(apod, maps)
        spec += cs.calc(maps)

    # Save spectra to HDF5 file
    outfile = ph2.spectra_file(args.simtype, args.field, args.year, args.nlat,
                               args.rlz[0], args.rlz[1], split_bands=args.split,
                               pbscaling=args.pbs)
    with h5py.File(outfile, 'w') as f:
        spec.to_hdf5(f)
