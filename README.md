## GOES Files

This repository contains a set of experiments to use GOES data. To run this code follow these steps using conda and pip packet managers

### Installation
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
### Usage
# Derived motion winds
The notebook [DerivedMotionWinds_cartopy.ipynb](https://github.com/rodhervan/GOES_Files/blob/main/DerivedMotionWinds_cartopy.ipynb) contains a function to create an image of the Derived Motion Winds vectors at different heights. By default it downloads images from the CMI band 14, and the DMW product which is available every hour. The input must be a date and time in the format YYYYMMDDhhmm and a destination folder.
