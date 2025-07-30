#!/bin/bash
#SBATCH -A mp107a
#SBATCH -C cpu
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 128  # Use full node
#SBATCH -q regular
#SBATCH --mail-user=vhoffman@lbl.gov
#SBATCH --mail-type=ALL

module load conda
conda activate s4bb

export LD_LIBRARY_PATH=$HOME/usr/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/reproduce_colin_results:$PYTHONPATH
export OMP_NUM_THREADS=1

mkdir -p mlsearch_bandpass_ghz

#With 128 cores and 101 tasks, each task gets 1 core

#Run all 101 GHz biases in parallel
for i in {0..100}; do
    ghz_bias=$(echo "scale=1; -5.0 + ${i} * 0.1" | bc)
    
    /usr/bin/time -v python -u phase2_mlsearch_ghz.py 2 20 --pbs --uniform-ghz-bias=${ghz_bias} --noffdiag=0 \
        &> jobs/ph2_mlsearch_f2_y20_n3_diag0_full_withpbs_uniform_${ghz_bias}GHz.out &
done

wait