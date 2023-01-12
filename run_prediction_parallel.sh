#!/bin/bash -l

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:0

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
module load gcc-libs/4.9.2
module load python3/recommended

echo "Loaded torch from local install"
. /home/ucbqc38/torch/install/bin/torch-activate

# Get the grid reference
echo $SGE_TASK_ID
gref=$(sed "${SGE_TASK_ID}q;d" grid_references.txt)
echo $gref

# Run the application
echo "Shell: Running python script"
#python3 -u train.py
module load python3/recommended
# Don't overwrite
#if [ ! -d 'results/03Masks/Green/${gref}' ] 
#then
    #python3 -u predict_from_best.py $gref;
#fi
# or overwrite
python3 -u predict_and_extract.py $gref config/best-predict-config.toml
# NB python directs to the wrong install of python!

cd /home/ucbqc38/Scratch/results
echo zip ${gref}.zip -q -r ./getmapping_*/$gref
zip ${gref}.zip -q -r ./getmapping_*/$gref

date
