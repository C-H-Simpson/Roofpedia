#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1
#$ -ac allow=EF 

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=48:00:0

# Request RAM (must be an integer followed by M, G, or T)
#$ -l mem=16G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=8G

# Set the name of the job.
#$ -N Predict

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucbqc38/Roofpedia_clean/Roofpedia_resample/

date

# Clean pip local env
#rm -r $HOME/.python3local/


## load the cuda module (in case you are running a CUDA program)
#echo "Shell: Loading CUDA modules"
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/10.1.243/gnu-4.9.2
module load torch-deps
##do-torch-install # fails
#
## Create the python environment
#module load python/3.7.4
## NB the pytorch build depends specifically on python 3.7.4
##module load pytorch/1.2.0/gpu
##module load opencv/3.4.1/gnu-4.9.2
#echo "Shell: installing pip dependencies"
module load python3/recommended
pip3 install --user geopandas webp mercantile==1.0.4 opencv-python geojson pygeos
pip3 install --user torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user toml

# Run the application
echo "Shell: Running python script"
#python3 -u train.py
python3 -u predict_from_best.py InnerLondon
# NB python directs to the wrong install of python!

# Make sure you have given enough time for the copy to complete!
date
