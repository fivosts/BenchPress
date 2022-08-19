#!/bin/bash
#SBATCH --job-name=basepretrain
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:8
#SBATCH -N 8
#SBATCH -c 10
#SBATCH --partition=learnfair,learnlab
#SBATCH --ntasks-per-node 8
#SBATCH --requeue
#SBATCH --constraint volta32gb
#SBATCH -e slurm_logs/pretrain_base-%j.err
#SBATCH -o slurm_logs/pretrain_base-%j.out
#SBATCH --mail-user=foivos@fb.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# Execute clgen command. Modify the command script accordingly.
srun bash ./benchpress --validate_per_epoch 0 --sample_per_epoch 0 --config model_zoo/BERT/bert_base_pretrain.pbtxt --workspace_dir /checkpoint/foivos/pretrained_corpus --local_filesystem /tmp/foivos/slurm_base_pretrained
srun rm -rf /tmp/foivos/slurm_base_pretrained
