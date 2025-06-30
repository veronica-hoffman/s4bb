# phase2_mlsearch.py
# ----------------
# ML search for phase 2 Chile optimization sims

import argparse
import numpy as np
import git
import h5py
import phase2_common as ph2

# Handle command line arguments
parser = argparse.ArgumentParser()
# Positional arguments
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
parser.add_argument('--noffdiag', type=int, default=1,
                    help='number of off-diagonal blocks to keep in BPCM')
parser.add_argument('--dry-run', action='store_true',
                    help='print configuration and exit')

# Starting guess for model parameters depends on which subfield we are looking
# at. These guesses come Colin's phase 1 results posting.
# https://cmb-s4.atlassian.net/wiki/spaces/XPI/pages/1149894714/Bischoff+Buza
def starting_guess(field):
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
    print(f'noffdiag = {args.noffdiag}')
    if args.dry_run:
        quit()
    
    # Code provenance
    #s4bbrepo = git.Repo('s4bb/')
    #print('s4bb version: {}'.format(s4bbrepo.head.object.hexsha))
    print('s4bb version: local files only (no git)')

    # Read CMB+fg+noise spectra
    data = ph2.get_spectra('comb', args.field, args.year, args.nlat, args.rlz[0], args.rlz[1],
                           split_bands=args.split, pbscaling=args.pbs)
    # Get likelihood data structure
    lik = ph2.get_likelihood(args.field, args.year, args.nlat, args.rlz[0], args.rlz[1],
                             args.split, args.pbs)

    # H-L precalculations
    # It seems to make a big difference to set noffdiag=1 for the
    # bandpower covariance matrix. Keeping two off-diagonal blocks
    # led to lots of failed mlsearch results.
    guess = starting_guess(args.field)
    lik.compute_fiducial_bpcm(lik.expv(guess, include_bias=False),
                              noffdiag=args.noffdiag, mask_noise=True)
    
    # Save results
    savefile = f'mlsearch/ph2_mlsearch_f{args.field}_y{args.year}_n{args.nlat}'
    if args.split:
        savefile += '_split'
    else:
        savefile += '_full'
    if args.pbs:
        savefile += '_withpbs'
    else:
        savefile += '_nopbs'
    savefile += '.npy'

    # Run maximum likelihood searches
    free = ['r', 'A_d','alpha_d', 'beta_d', 'A_s', 'alpha_s', 'beta_s', 'epsilon',
            'Delta_d', 'Delta_s'] 
    limits = {'beta_d': [1.0, 2.0], 'alpha_d': [-2.0, 0.5],
              'beta_s': [-4.0, -2.0], 'alpha_s': [-2.0, 0.5],
              'epsilon': [-1,1], 'Delta_d': [0.5,1.1], 'Delta_s': [0.5,1.1]}
    method = 'L-BFGS-B'
    options = {'maxls': 30}
    x = np.zeros(shape=(12,data.shape[2]))
    for i in range(data.shape[2]):
        print(i)
        (result, fval, status) = lik.mlsearch(data[:,:,i], guess, free=free, limits=limits,
                                              method=method, options=options)
        x[0,i] = status
        x[1,i] = fval
        x[2,i] = result['r']
        x[3,i] = result['A_d']
        x[4,i] = result['alpha_d']
        x[5,i] = result['beta_d']
        x[6,i] = result['A_s']
        x[7,i] = result['alpha_s']
        x[8,i] = result['beta_s']
        x[9,i] = result['epsilon']
        x[10,i] = result['Delta_d']
        x[11,i] = result['Delta_s']
    # Done, save results.
    np.save(savefile, x)
