#!/bin/bash
#SBATCH --job-name=basefeat
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:8
#SBATCH -N 8
#SBATCH -c 10
#SBATCH --partition=learnfair,learnlab
#SBATCH --ntasks-per-node 8
#SBATCH --requeue
#SBATCH -e slurm_logs/base_opencl_head_features_slurm-%j.err
#SBATCH -o slurm_logs/base_opencl_head_features_slurm-%j.out
#SBATCH --mail-user=foivos@fb.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# Execute clgen command. Modify the command script accordingly.
srun rm -rf /scratch/tmp_foivos/features_base_opencl
srun bash ./clgen --print_samples=False --sample_indices_limit 100 --only_sample --start_from_cached --sample_workload_size 256 --sample_per_epoch 0 --config model_zoo/BERT/base_opencl_features.pbtxt --workspace_dir /checkpoint/foivos/base_opencl_features # --local_filesystem /scratch/tmp_foivos/features_base_opencl
srun rm -rf /scratch/tmp_foivos/features_base_opencl
