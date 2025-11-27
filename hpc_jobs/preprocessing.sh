#!/bin/sh

#BSUB -J prec_pipeline

#BSUB -q hpc

#BSUB -n 32

#BSUB -R "rusage[mem=4GB]"

#BSUB -W 2:00

#BSUB -B

#BSUB -o hpc_jobs/logs/Output_%J.out
#BSUB -e hpc_jobs/logs/Output_%J.err

cd ~/AIS-MDA

. .venv/bin/activate

python -m src.preprocessing.csv2pkl \
    --input_dir /dtu/blackhole/10/178320/ \
    --output_dir /dtu/blackhole/10/178320/preprocessed_3/pickle/ \
    --cargo_tankers_only \
    --run_name pipeline_3_csv2pkl

python -m src.preprocessing.map_reduce \
    --input_dir /dtu/blackhole/10/178320/preprocessed_3/pickle/ \
    --output_dir /dtu/blackhole/10/178320/preprocessed_3/final/ \
    --num_workers 32 \
    --run_name pipeline_3_map_reduce

python -m src.preprocessing.train_test_split \
    --data_dir /dtu/blackhole/10/178320/preprocessed_3/final/ \
    --val_size 0.1 \
    --test_size 0.1