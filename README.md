<p align="center">
  <a href="https://ual.sg/">
    <img src="images/logo.jpg" alt="Logo">
  </a>
  <h3 align="center">Roofpedia - Mapping Roofscapes with AI</h3>
  <p align="center">
    <br />
    <a href="https://ual.sg/project/roofpedia/"><strong>Explore Sustainable Roofscapes Around the World »</strong></a>
  </p>
</p>

This is the official repo of Roofpedia, an open registry of green roofs and solar roofs across the globe identified by Roofpedia through deep learning.

In this repo you will find:
* A ready to use dataset of 1,812 manually labelled polygons of rooftop greenery and solar panels covering 8 cities.
* A ready to use dataset of building footprints identified with Green Roofs and/or Solar roofs by Roofpedia. Download data and labels [here](https://drive.google.com/file/d/183b23u6hXqdU2LmDOdCYcmNa907A1XA8/view?usp=sharing)
* A straight forward pipeline to run prediction on your own satellite image dataset
* A guide on how you can tweak the pipeline to detect and tag roof features to OSM building footprints (coming up)

## Running Roofpedia
Steps:
1. Install prequisites
2. Download and extract weights and sample dataset
3. run predict_and_extract.py
4. get result!
### 1. Prerequisites

You could use `environment.yml` to create a conda environment for Roofpedia

  ```sh
  conda env create -f environment.yml
  ```

For non-gpu users, use `environment_cpu.yml` instead.

  ```sh
  conda env create -f environment_cpu.yml
  ```
### 2. Data Preparation

Download the pretrained weights and sample dataset [here](https://drive.google.com/file/d/1uRsuXxSEhDEHaa8CoMmncpbClJ2fapJx/view?usp=sharing) and extract them to the root folder of the repo.

For custom inference, datasets should be processed and placed in the `results` folder. See more details in later sections.


### Prediction

Predictions can be carried out by running the following sample code. The name of the city depends on the name of each dataset.

```sh
  python predict_and_extract.py <city_name> <type>
```

A sample dataset is provided in the results folder with the name `NY` for prediction, just run

```sh
  python predict_and_extract.py NY Solar
```

for Greenroofs, run

```sh
  python predict_and_extract.py NY Green
```

See the result in `NY_Solar.geojson` or `NY_Green.geojson` in `04Result` folder and visualise the results in QGIS or ArcGIS.

### Custom Dataset
Custom Dataset pairs can be created with QGIS using tiling functions.
1. Create a WMTS satellite tile connection with any WMTS server. You can use Mapbox's WMTS server for good quality images.
2. With QuickOSM, query and download the building footprint of a desired area for prediction.
3. Save the building polygons to `01City` folder.
4. Callup QGIS toolbar (`Ctrl + Alt +T`), in `Raster Tools`, choose `Generate XYZ Tiles(Directory)` to generate satellite tiles for the area by using Canvas Extent. Use Zoom 19 and save to `02Images/Cityname`. Make sure the QGIS project CRS is "EPSG:4326" when you do this.
5. Check that `config/predict_config.toml` points to the city, target, and checkpoint that you want to use for prediction.

A unified script in extracting building polygons and downloading satellite tiles from Mapbox is a work-in-progress.
### Custom Dataset File Structure
The structure of the `results` folder is as follows:

📂results  
 ┣ 📂01City   
 ┃- ┗ 📊Cityname1.geojson  
 ┃- ┗ 📊Cityname2.geojson  
 ┣ 📂02Images  
 ┃--- ┗ 📂Cityname1  
 ┃--- ┗ 📂Cityname2  
 ┣ 📂03Masks  
 ┃--- ┗ 📂Green  
 ┃---   ┗ 📂Cityname1  
 ┃---   ┗ 📂Cityname2  
 ┃--- ┗ 📂Solar  
 ┃---   ┗ 📂Cityname1  
 ┃---   ┗ 📂Cityname2  
 ┣ 📂04Results  
 ┃- ┗ 📊Cityname1_Green.geojson  
 ┃- ┗ 📊Cityname1_Solar.geojson  
 ┃- ┗ 📊Cityname2_Green.geojson  
 ┃- ┗ 📊Cityname2_Solar.geojson  

`01City` contains geojson files of building polygons  
`02Images` contains a slippymap directory of satellite images. For the pre-trained models, a zoom level of 19 is required.  
`03Masks` contains predicted masks of each tile according to object type
`04Results` contains final cleaned building footprints tagged with the specific object type


### Training

1. Produce labels in QGIS. Export using "Generate XYZ Tile (Directory)" as white polygons on a black background, making sure the QGIS project CRS is set to "EPSG:4326". Put in `dataset/labels/`. The labels are not limited to green roof or solar panels, but can be any custom object on the roof as long as sufficient labels are provided.
2. Get imagery and export in QGIS with same method. Put in `dataset/images/`.
3. Run `dataset.py`. This will remove images in which nothing is labelled, and do a train/test split. Visually check the resulting image labels make sense.
4. Check `config/train-config.toml` identifies the data you are training with. If you want to continue from a pre-trained model, set the 'model_path' to the path of the pre-trained model.
5. Run `train.py`. A new model checkpoint will have been created that you can then use for prediction.


## Paper

A [paper](https://doi.org/10.1016/j.landurbplan.2021.104167) about the work was published in _Landscape and Urban Planning_ and it is available open access.

If you use this work in a scientific context, please cite this article.

Wu AN, Biljecki F (2021): Roofpedia: Automatic mapping of green and solar roofs for an open roofscape registry and evaluation of urban sustainability. Landscape and Urban Planning 214: 104167, 2021. doi:10.1016/j.landurbplan.2021.104167

```
@article{roofpedia,
  author = {Abraham Noah Wu and Filip Biljecki},
  doi = {10.1016/j.landurbplan.2021.104167},
  journal = {Landscape and Urban Planning},
  pages = {104167},
  title = {Roofpedia: Automatic mapping of green and solar roofs for an open roofscape registry and evaluation of urban sustainability},
  url = {https://doi.org/10.1016/j.landurbplan.2021.104167},
  volume = {214},
  year = 2021
}
```


## Limitations, issues, and future work

Roofpedia is an experimental research prototype, which leaves much opportunity for improvement and future work.

As with all other machine learning workflows, the results are not always 100% accurate.
Much of the performance of the predictions (e.g. classification of whether a building has a solar panel on its rooftop) depends on the quality of the input imagery.
Therefore, some buildings are misclassified, especially in imagery in which it is difficult even for humans to discern rooftop greenery and photovoltaics, resulting in false positives and false negatives.
However, when these results are aggregated at the city-scale, the results tend to be more accurate.

For future work, we hope to add more cities to our collection and add the temporal aspect to the project, tracking the evolution of greenery and solar panels through time.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


## Contact

[Abraham Noah Wu](https://ual.sg/authors/abraham/), [Urban Analytics Lab](https://ual.sg), National University of Singapore, Singapore


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

Roofpedia is made possible by using the following packages

* [PyTorch](https://pytorch.org/)
* [GeoPandas](https://geopandas.org/)
* [Robosat](https://github.com/mapbox/robosat) -
loading of slippy map tiles for training and mask to feature function is borrowed from robosat

This research is part of the project Large-scale 3D Geospatial Data for Urban Analytics, which is supported by the National University of Singapore under the Start-Up Grant R-295-000-171-133.

We gratefully acknowledge the sources of the used input data.

Some of the aspects of the project and its name - Roofpedia - are inspired by [Treepedia](http://senseable.mit.edu/treepedia), an excellent project by the [MIT Senseable City Lab](https://senseable.mit.edu) to measure and map the amount of street greenery in cities from the pedestrian perspective, and compare cities around the world.
