#!/bin/bash
#SBATCH -A mp107a
#SBATCH -C cpu
#SBATCH -t 6:00:00
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -q regular

# Run maximum likelihood search for bandpass miscalibration test
# 6 bands (HF-2, HF-1, MF-1, MF-2, LF-1, LF-2) × 2 bias directions (+/-2%) = 12 tasks
# Each task runs serially over 100 realizations
# Set logical cpus per task = 2 * 128 / 12 ≈ 21, but using 10 to be conservative

module load conda
conda activate s4bb

export LD_LIBRARY_PATH=$HOME/usr/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/reproduce_colin_results:$PYTHONPATH
export OMP_NUM_THREADS=10

mkdir -p jobs

# Define the bands
bands=("HF-2" "HF-1" "MF-1" "MF-2" "LF-1" "LF-2")

# Run bandpass miscalibration tests
for band in "${bands[@]}"; do
    # +2% bias
    /usr/bin/time -v python -u phase2_mlsearch.py 1 20 --pbs --bias-band="${band}" --bias-percent=2.0 --noffdiag=0 \
        &> jobs/ph2_mlsearch_f1_y20_${band}_plus2pct.out &
    
    # -2% bias  
    /usr/bin/time -v python -u phase2_mlsearch.py 1 20 --pbs --bias-band="${band}" --bias-percent=-2.0 --noffdiag=0 \
        &> jobs/ph2_mlsearch_f1_y20_${band}_minus2pct.out &
done

wait