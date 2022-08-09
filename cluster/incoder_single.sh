#!/bin/bash
#SBATCH --job-name=incoder_rodinia_grewe
#SBATCH --time=72:00:00
#SBATCH -c 10
#SBATCH --mem=350G
#SBATCH --gres=gpu:8
#SBATCH -N 8
#SBATCH --ntasks-per-node=8
#SBATCH --partition=learnfair,learnlab
#SBATCH --constraint volta32gb
#SBATCH -e slurm_logs/incoder_slurm-%j.err
#SBATCH -o slurm_logs/incoder_slurm-%j.out

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# Execute clgen command. Modify the command script accordingly.
srun bash ./clgen --evaluate_candidates --start_from_cached --print_samples=False --custom_incoder_ckpt $1 --override_preprocessing --override_encoding --config model_zoo/BERT/incoder.pbtxt --workspace_dir /checkpoint/foivos/incoder_workspace_grewe --local_filesystem /tmp/foivos/slurm_$j --sample_workload_size 128 --skip_first_queue # 1024
srun rm -rf /tmp/foivos/slurm_$j
