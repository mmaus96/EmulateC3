#!/bin/bash
#SBATCH -p regular
#SBATCH -A cosmosim
#SBATCH -t 4:00:00
#SBATCH -J train_emu
#SBATCH -o train_emu
#SBATCH --mail-type=All
#SBATCH --mail-user  joe.derose13@gmail.com
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -L SCRATCH
#SBATCH --exclusive

#call salloc -N 1 -A cosmosim -q interactive -t 4:00:00 -C haswell -L SCRATCH to get an interactive node for training

module load python
conda activate train_env

export PYTHONPATH=${PYTHONPATH}:/pscratch/sd/m/mmaus/DESI_velocileptors_redef/emulator/FM/NeuralNet #replace with your cobayalss directories
export PYTHONPATH=${PYTHONPATH}:/global/homes/m/mmaus/Python/velocileptors
export HDF5_USE_FILE_LOCKING=FALSE

#need to reformat the training data file before running training
srun -n 1 python reformat_training_data_pkell.py fs_abacus_am_wcdm_theory_LRG2_direct.yaml #Generate_training_data.yaml

module load tensorflow

srun -n 1 python -u train_nn_emu.py configs/train_p0_scan_4_128_1.yaml
srun -n 1 python train_nn_emu.py configs/train_p2_scan_4_128_1.yaml
srun -n 1 python train_nn_emu.py configs/train_p4_scan_4_128_1.yaml

