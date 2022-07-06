#!/bin/bash -l

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=00:05:0

# Request RAM (must be an integer followed by M, G, or T)
#$ -l mem=16G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=8G

# Setup job array.
#$ -t 1-30

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucbqc38/Roofpedia_clean/Roofpedia_resample/

date

echo "Loading recommended python"
module load python3/recommended

echo "Loaded torch from local install"
. /home/ucbqc38/torch/install/bin/torch-activate

# Get the grid reference
gref=$(sed "${SGE_TASK_ID}q;d" grid_references.txt)
echo $gref

# Run the application
echo "Shell: Running python script"
#python3 -u train.py
echo python3 -u predict_from_best.py $gref
# NB python directs to the wrong install of python!

date
