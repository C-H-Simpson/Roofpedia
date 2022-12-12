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

#$ -e $ERRFILE
#$ -o $OUTFILE

date

module load gcc-libs/4.9.2

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/ucbqc38/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ucbqc38/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ucbqc38/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ucbqc38/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate ../env

pwd

# Run the application
python -u prepare_imagery_from_files.py -g $gref10k -i $imagery_dir -o $destination -L $labels

date
