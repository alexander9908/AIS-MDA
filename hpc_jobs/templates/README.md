# How to run a job on the HPC

## Setup Weights and Biases
Reinstall environment **on the HPC** to ensure `wandb` is installed
```bash
pip install -r env/requirements.txt
```

Login to Wandb:
```bash
wandb login
```

## Create a batch job script
See example files [cpu_job.sh.example](hpc_jobs/templates/cpu_job.sh.example) and [gpu_job.sh.example](hpc_jobs/templates/gpu_job.sh.example) for a CPU and GPU job respectively

Optionally update resource requirements.

Add run commands in the bottom, i.e.
```shell
#!/bin/sh
#BSUB -J test_run
#BSUB -q hpc
### Rest of resource commands

cd ~/AIS-MDA ### Ensure correct path
. .venv/bin/activate ### Ensure venv is called .venv

python -m src.train.train_traj \
    --run_name test_run \
    --lr 0.0003 \
    ### etc., or:

### using config file
python -m src.train.train_traj --config path/to/config.yaml 
```

## Submit batch job
To submit run:
```bash
bsub < hpc_jobs/job_script_name.sh
```

To check queue status run:
```bash
bstat
```

Too end job prematurely run
```bash
bkill [JOBID]
```

## See run in Wandb
Once job is running it will show up at the [AIS-MDA project in WandB](https://wandb.ai/bertram-hage-danmarks-tekniske-universitet-dtu/AIS-MDA)