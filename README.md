# GOES Files

This repository contains a set of notebooks and python files to handle GOES data. To run this code follow these steps using conda and pip packet managers

## Installation
1. Navigate to the folder where the project files are going to be:
```bash
   cd your/path/to/project
```
2. Clone this repository
```bash
   git clone https://github.com/rodhervan/GOES_Files.git
```
3. Create a virtual environment using conda
```bash
   conda create -n Goes_env -c conda-forge gdal libgdal-hdf5 pygrib
```
4. Activate the virtual environment
```bash
   conda activate GOES_env
```
5. Install remaining packages using PIP
```bash
   pip install -r requirements.txt
```
## Usage
### Derived motion winds
The notebook [DerivedMotionWinds_cartopy.ipynb](https://github.com/rodhervan/GOES_Files/blob/main/DerivedMotionWinds_cartopy.ipynb) contains a function to create an image of the Derived Motion Winds vectors at different heights. By default it downloads images from the CMI band 14, and the DMW product which is available every hour.

To do this run the funtion `plot_dmw`. The inputs must for this function must be the time in the format YYYYMMDDhhmm and a destination folder, since the data is only available hourly the hh field must be a number from 0 to 24 and the mm number must always be set to 00. The reulting image will be a plot of the CMI band 14 as background, sliced at a set extent and the velocity vectors divided by heights (according to pressure).

### Files Downloader
This notebook can be used to download GOES ´.nc´ files of different products. By default it is set to slice the images at a set value of `x=(-0.05, 0.07)` and  `y=(0.09, -0.03)`, and execute a downscale to a resolution of 300x300 pixels.

### Correlation of variables
To justify the usage of these images as a prediction input, a correlation study between the GHI and the CMI was done at different time lags. This process can be seen in the notebook [GOES_corr_B02 processing.ipynb](https://github.com/rodhervan/GOES_Files/blob/main/GOES_corr_B02%20processing.ipynb).

### Convolutional Neural Network for GOES Images
For this process two notebooks are used. The first one [cnn_B02_cos_sin 2H_train.ipynb](https://github.com/rodhervan/GOES_Files/blob/main/cnn_B02_cos_sin%202H_train.ipynb) is related to loading and handling of data, to train a CNN model. The second notebook [cnn_B02_cos_sin 2H_Inference.ipynb](https://github.com/rodhervan/GOES_Files/blob/main/cnn_B02_cos_sin%202H_Inference.ipynb) continues this process to load the model, and execute inferences on any given image input and matching GHI data. 

