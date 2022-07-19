# An Open-Source Automatic Survey of Green Roofs in London using Segmentation of Aerial Imagery 
This repo contains code used to extract the location and area of green roofs in London from aerial imagery.

## A note on this fork
This is a fork of Roofpedia, an open registry of green roofs and solar roofs across the globe identified by Roofpedia through deep learning.
This fork is the result of our experiments trying to improve the performance of Roofpedia.
If you aren't familiar with Roofpedia, we recommend looking at the [main repo](https://github.com/ualsg/Roofpedia) first.

## Differences from the ualsg repo
* Early stopping
* Inclusion of imagery tiles that do not contain any green roof. This leads to better performance (fewer false positives). The background is over-sampled to avoid this creating problems for gradient descent.
* Experimentation with augmentation methods. We found that adding in random augmentations to the sharpness of the imagery improved performance slightly.
* We have a method of parallel prediction using a large number of CPUs rather than a GPU.


## Training process
1. Install prerequisites from `environment.yml`.
2. Prepare data (see below).
3. Run `dataset.py` to apply a train test split.
4. Run experiments.py. This will produce a large number of directories with the pattern `experiment_{timestamp}`. Each of these has the results of one training experiment. Training will require a GPU.
5. Select the best experiment, and manually set the name of the directory and checkpoint in `predict_from_best.py`.

## Prediction process
1. Prepare data.
2. If you are going to run parallel, divide the domain up into gridsquares using `src/construct_gridreferences.py`.
3. Run `predict_from_best.py`, either parallel or with a GPU. The script for running in parallel on UCL Myriad `run_prediction_parallel.sh` may be informative. It runs one job for each grid reference square, but does not require a GPU.

## Data preparation
This version of the repo has additional requirements for data.
1. Labelled polygons for training. Produce these by drawing polygons in QGIS. Export these as slippymap tiles from QGIS using GenerateXYZ.
2. Imagery. Export this as slippymap tiles from QGIS using GenerateXYZ. 
3. Training area demarcation. This should be a vector file marking out all the areas what were labelled. This is so that areas that have been inspected but do not contain green roof can be used for training. This dataset is uesed in `dataset.py`.
