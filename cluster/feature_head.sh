#!/bin/bash
#SBATCH --job-name=512_miniscule_features
#SBATCH --time=72:00:00
#SBATCH -c 10
#SBATCH --gpus-per-node=8
#SBATCH -N 8
#SBATCH --partition=learnfair,learnlab
#SBATCH --ntasks-per-node=8
#SBATCH --requeue
#SBATCH -e slurm_logs/feature_head_slurm-%j.err
#SBATCH -o slurm_logs/feature_head_slurm-%j.out
#SBATCH --mail-user=foivos@fb.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

echo "This model has been completed for all 3 feature spaces against rodinia."

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# Execute clgen command. Modify the command script accordingly.
srun rm -rf /tmp/foivos/feature_head_tiny
srun bash ./clgen --start_from_cached --print_samples=False --sample_workload_size 512 --only_sample  --sample_per_epoch 0 --config model_zoo/BERT/feature_encoding.pbtxt --workspace_dir /checkpoint/foivos/feature_head --local_filesystem /tmp/foivos/feature_head_tiny
srun rm -rf /tmp/foivos/feature_head_tiny
