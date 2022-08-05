#!/bin/bash

########################
# To start a Distributed slurm node session type
# - salloc --time=HH:MM:SS --gres=gpu:<X> -c <Y> --ntasks-per-node=<X> -N <Z> --nodelist=N1,N2,... bash -l
# Where:
# X: number of GPUs per node
# Y: Number of CPUs per node
# Z: Number of Nodes
# --nodelist is optional to target specific nodes.
#
#
# Then, within the allocated salloc session execute this script.

# Collect the address of the master node.
export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

# Execute benchpress command. Modify the command script accordingly.
srun ./ddp_run.sh

