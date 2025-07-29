#!/bin/bash -l
#SBATCH -J training_dat
#SBATCH -t 02:00:00
#SBATCH -N 4
#SBATCH -o output_logs/job%j.out
#SBATCH -e output_logs/job%j.err
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -A desi


module load python
conda activate train_env
# export PYTHONPATH=${PYTHONPATH}:/pscratch/sd/m/mmaus/DESI_velocileptors_redef/emulator/FM/NeuralNet #replace with youre cobayalss 
# export PYTHONPATH=${PYTHONPATH}:/global/homes/m/mmaus/Python/velocileptors

export HDF5_USE_FILE_LOCKING=FALSE



mpirun -np 500 python generate_training_data.py fs_abacus_am_wcdm_theory_BGS_direct.yaml
# mpirun -np 500 python generate_training_data.py fs_abacus_am_wcdm_theory_LRG1_direct.yaml
# mpirun -np 500 python generate_training_data.py fs_abacus_am_wcdm_theory_LRG2_direct.yaml
# mpirun -np 500 python generate_training_data.py fs_abacus_am_wcdm_theory_LRG3_direct.yaml
# mpirun -np 500 python generate_training_data.py fs_abacus_am_wcdm_theory_ELG1_direct.yaml
# mpirun -np 500 python generate_training_data.py fs_abacus_am_wcdm_theory_ELG2_direct.yaml
# mpirun -np 500 python generate_training_data.py fs_abacus_am_wcdm_theory_QSO_direct.yaml