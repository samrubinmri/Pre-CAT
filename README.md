# Pre-CAT <img src="https://i.ibb.co/gMQ7MCb/Subject-4.png" width="50">

## Welcome to Pre-CAT 

**Pre-CAT** is a preclinical CEST-MRI data analysis toolbox and Streamlit webapp. **Pre-CAT** was designed for processing cardiac and abdominal CEST acqisitions using the radial FLASH sequence described by Weigand-Whittier et al., but has been expanded for use with all ParaVision 6/7 CEST acqisitions.

**Pre-CAT** processes and displays Z-spectra, Lorentzian difference plots, pixelwise maps, and field maps. Processed data is also saved in organized pickle files for downstream tasks.

## Requirements

A pre-prepared Conda environment is available in the `environment.yml` file. To install the environment, run:

```sh
conda env create -f environment.yml
conda activate pre-cat
python -m pip list
```

In addition to the included environment, [`streamlit-drawable-canvas`](https://github.com/andfanilo/streamlit-drawable-canvas) must be installed separately:

```sh
pip install streamlit-drawable-canvas 
```

Finally, [`BART`](https://mrirecon.github.io/bart/) is also required. Please follow the instructions in the **README** file for installation.

> ⚠ **Note for Mac Users:**  
> If you're using an **M1 Mac**, please follow the **MacPorts** version of the installation instructions for [`BART`](https://mrirecon.github.io/bart/).

## Instructions

In-depth instructions are included in the Streamlit interface. On release, a video tutorial will be included here as well.
