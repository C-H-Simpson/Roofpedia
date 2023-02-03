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
* We have a method of parallel prediction using a large number of CPUs rather than a GPU. Currently this relies on UK grid references, but can be modified for another CRS. (GPU still required for training)
* K-fold cross validation of the confusion matrix.
* We do not use the OpenStreetMap SlippyMap convention for tiling within the segmentation pipeline. Instead we use the native CRS of the input raster (OSGB36/EPSG27700) which we slice into 256x256 pixel images. This means changes would need to be made to the tiling in order for this to work outside the UK.
* We use rasterio for creating the polygons from the raster rather than purpose-written code, and gdal for tiling the imagery rather than QGIS.
* We do not use morphological operations to denoise the results. Instead we imposed a minimum polygon size on the final predictions.

## Ideas to take away to your own project
* Try including tiles that don't contain the signal class in your training sample, but undersample them. It is very important to include these negative examples in the validation dataset.
* Try the albumentations library, as it makes it easy to try out different data augmentations.
* Try a different set of imagery.

## Data preparation
This version of the repo has additional requirements for data.
0. Install prerequisites from `environment.yml`.
1. Labelling. Labelling was performed in QGIS to produce a geojson vector layer. The tiling routine then processes this into a raster with the same resolution and CRS as the imagery. You will also need to identify the limits of the area you have labelled but which do not contain the feature you want to identify, as the negative examples are crucial.
2. Tiling. A major change from the UALSG/Roofedia repo is that we had all of the raster data downloaded rather than in a WMS. We therefore directly tile the raster rather than using QGIS. To do this, we used the script prepare_imagery_from_tiles.py.
3. Split. We implemented k-fold testing. The split is performed by the script dataset.py.
4. Experimentation. experiment.py runs several experiments to determine a good configuration for training.
5. Choosing the best configuration. After running experiment.py, the script plot_experiments.py determines which was the best performing and copies it into config/best-predict-config.toml.
6. K-fold. After identifying the best configuration, training of the k-fold models is performed by kfold_testing.py.
7. Erosion. You may want to experiment with erosion etc. using erosion_experiment.py. These settings will need to be manually changed downstream.
8. Validation. After the k models have been trained, predict_validation_from_best_kfold.py will create confusion matrices.
9. Full scale prediction. predict_and_extract.py either parallel or with a GPU. The script for running in parallel on UCL Myriad `run_prediction_parallel.sh` may be informative. It runs one job for each grid reference square, but does not require a GPU.
