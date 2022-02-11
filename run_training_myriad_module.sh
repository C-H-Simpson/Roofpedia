#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1
#$ -ac allow=EF 

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:0

# Request RAM (must be an integer followed by M, G, or T)
#$ -l mem=8G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=0G

# Set the name of the job.
#$ -N GPUJob_module

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID :)
#$ -wd /home/ucbqc38/Scratch/Roofpedia/

date

# Clean pip local env
rm -r $HOME/.python3local/


# load the cuda module (in case you are running a CUDA program)
echo "Shell: Loading CUDA modules"
module unload compilers mpi
module load compilers/gnu/4.9.2
module load cuda/10.1.243/gnu-4.9.2
module load torch-deps
#do-torch-install # fails

# Create the python environment
#module load python/3.7.4
# NB the pytorch build depends specifically on python 3.7.4
#module load pytorch/1.2.0/gpu
#module load opencv/3.4.1/gnu-4.9.2
echo "Shell: installing pip dependencies"
module load python3/recommended
pip3 install --user geopandas webp mercantile==1.0.4 opencv-python geojson pygeos
pip install --user torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Run the application
echo "Shell: Running python script"
echo "Shell: Training"
python3 -u train.py
#echo "Shell: Predicting"
#python3 -u predict_and_extract.py LondonInner Green
# NB python directs to the wrong install of python!

date
