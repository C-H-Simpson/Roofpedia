#!/bin/bash -l

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:0

# Request RAM (must be an integer followed by M, G, or T)
#$ -l mem=8G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=8G

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucbqc38/Roofpedia_clean/Roofpedia_resample/

date

module load gcc-libs/4.9.2

conda activate ../env

# Run the application
python -u prepare_imagery_from_files.py -g $gref10k -o $destination -L $labels

date
