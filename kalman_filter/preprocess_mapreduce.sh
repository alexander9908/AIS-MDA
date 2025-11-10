#!/bin/sh

### Job Name:
#BSUB -J mapreduce_preprocess

### Queue Name:
#BSUB -q hpc

### Requesting 16 CPU cores, 16GB memory per core (MapReduce is memory-intensive)
#BSUB -n 16
#BSUB -R "rusage[mem=16GB]"

### Setting a runtime limit of 24 hours (conservative for 3 months of data)
#BSUB -W 24:00

### Email notification when job begins and ends
#BSUB -B
#BSUB -N

### Output and error files
#BSUB -o hpc_jobs/logs/mapreduce_%J.out
#BSUB -e hpc_jobs/logs/mapreduce_%J.err


### cd to repo dir
cd ~/AIS-MDA

### activate environment
. .venv/bin/activate

### load python module
module swap python3/3.13.2

### Define paths - UPDATE THESE for your HPC setup
INPUT_DIR="/dtu/blackhole/10/178320/preprocessed_1/pickles"
TEMP_DIR="/work3/s204572/AIS-data/map_reduce_temp"
FINAL_DIR="/work3/s204572/AIS-data/map_reduce_final"

### Create output directories if they don't exist
mkdir -p $TEMP_DIR
mkdir -p $FINAL_DIR

echo "========================================="
echo "MapReduce Preprocessing for AIS Data"
echo "========================================="
echo "Input directory:  $INPUT_DIR"
echo "Temp directory:   $TEMP_DIR"
echo "Final directory:  $FINAL_DIR"
echo "Start time:       $(date)"
echo "========================================="

### Count input files
NUM_FILES=$(ls -1 $INPUT_DIR/*.pkl | wc -l)
echo "Found $NUM_FILES pickle files to process"
echo ""

### Run MapReduce preprocessing
python -m src.preprocessing.map_reduce \
    --input_dir $INPUT_DIR \
    --temp_dir $TEMP_DIR \
    --final_dir $FINAL_DIR \
    --n_workers 16

echo ""
echo "========================================="
echo "MapReduce Complete!"
echo "End time: $(date)"
echo "========================================="

### Count output files
NUM_OUTPUT=$(ls -1 $FINAL_DIR/*_processed.pkl 2>/dev/null | wc -l)
echo "Created $NUM_OUTPUT processed trajectory files"
echo ""
echo "Output location: $FINAL_DIR"
echo "Ready for Kalman Filter evaluation!"
echo ""
echo "Next step: bsub < hpc_jobs/eval_kalman_full.sh"
