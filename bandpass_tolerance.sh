#!/bin/bash
#SBATCH -A mp107a
#SBATCH -C cpu
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -q regular
#SBATCH --mail-user=vhoffman@lbl.gov
#SBATCH --mail-type=ALL

# Run bandpass tolerance search for all bands
# 6 bands (HF-2, HF-1, MF-1, MF-2, LF-1, LF-2) = 6 tasks
# Each task runs serially over 10 realizations

module load conda
conda activate s4bb

export LD_LIBRARY_PATH=$HOME/usr/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/reproduce_colin_results:$PYTHONPATH
export OMP_NUM_THREADS=10

# Define the bands
bands=("HF-2" "HF-1" "MF-1" "MF-2" "LF-1" "LF-2")

# Run bandpass tolerance tests
for band in "${bands[@]}"; do
    /usr/bin/time -v python -u bandpass_tolerance.py 2 20 "${band}" --unbiased-file mlsearch/ph2_mlsearch_f2_y20_n3_diag0_full_withpbs.npy --pbs --r-bias-tol 0.0001 --n-realizations 10  \
        &> jobs/ph2_tolerance_f2_y20_n3_diag0_full_withpbs_${band}_rtol1e-04_nrlz10.out &
        
done

wait