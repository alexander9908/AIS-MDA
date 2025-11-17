#!/bin/sh
#
# TPTrans full 3-month training on DTU A100 80GB
#

### LSF options --------------------------------------------------------------
### -- specify queue (A100 GPUs) --
#BSUB -q gpua100

### -- set the job Name --
#BSUB -J tptrans_full_3m

### -- ask for number of CPU cores (min 4 per GPU is recommended) --
#BSUB -n 4

### -- select resources: 1 GPU in exclusive process mode, 80GB card --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[gpu80gb]"

### -- set walltime limit: hh:mm (max 24:00 for GPU queues) --
#BSUB -W 24:00

### -- request system memory (per job, not per core) --
#BSUB -R "rusage[mem=8GB]"

### -- (optional) email for notifications --
##BSUB -u s221221@dtu.dk

### -- send notification at start and completion --
#BSUB -B
#BSUB -N

### -- output and error files. %J is the job-id --
#BSUB -o hpc_jobs/logs/tptrans_full_%J.out
#BSUB -e hpc_jobs/logs/tptrans_full_%J.err

### -- end of LSF options ----------------------------------------------------

# Show which GPU/node we got (useful for debugging)
nvidia-smi

# Go to project root
cd ~/AIS-MDA

# Activate virtual environment
. .venv/bin/activate

# Load / swap Python & CUDA modules (adjust if your environment differs)
module swap python3/3.13.2
module swap cuda/12.6.3

# Run the TPTrans training with the full config
# (make sure configs/traj_tptrans_full.yaml exists and is correct)
python src/train/train_traj_ES.py --config_path configs/traj_tptrans_full.yaml