#!/bin/bash
#SBATCH --partition=special_cs
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=4        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:1             # Using 1 GPU card
#SBATCH --mem=64GB               # Request 50GB memory
#SBATCH --time=0-23:59:00        # Time limit day-hrs:min:sec
#SBATCH --output=output.log   # Standard output
#SBATCH --error=error.err    # Standard error log

pwd
nvidia-smi
python train.py
python infer.py