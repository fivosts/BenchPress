#!/bin/bash
#SBATCH --job-name=base_opencl
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:8
#SBATCH -N 8
#SBATCH -c 10
#SBATCH --partition=learnfair,learnlab
#SBATCH --ntasks-per-node 8
#SBATCH --requeue
#SBATCH -e slurm_logs/base_opencl_head_slurm-%j.err
#SBATCH -o slurm_logs/base_opencl_head_slurm-%j.out
#SBATCH --mail-user=foivos@fb.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# Execute clgen command. Modify the command script accordingly.
srun bash ./clgen --only_sample --sample_indices_limit 100 --print_samples=False --start_from_cached --sample_workload_size 256 --sample_per_epoch 0 --config model_zoo/BERT/base_opencl.pbtxt --workspace_dir /checkpoint/foivos/base_opencl --local_filesystem /tmp/foivos/slurm_$j
srun rm -rf /tmp/foivos/slurm_$j
