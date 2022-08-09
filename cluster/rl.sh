#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:8
#SBATCH -N 2
#SBATCH -c 10
#SBATCH --partition=learnfair,learnlab
#SBATCH --ntasks-per-node 8
#SBATCH --requeue
#SBATCH -e slurm_logs/rl-%j.err
#SBATCH -o slurm_logs/rl-%j.out
#SBATCH --mail-user=foivos@fb.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# Execute clgen command. Modify the command script accordingly.
srun bash ./benchpress --validate_per_epoch 0 --sample_per_epoch 3 --config model_zoo/BERT/rl_base_opencl.pbtxt --workspace_dir /checkpoint/foivos/reinforcement_learning --local_filesystem /tmp/foivos/slurm_reinforcement_learning
srun rm -rf /tmp/foivos/slurm_reinforcement_learning
