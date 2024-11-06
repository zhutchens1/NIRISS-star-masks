# JWST NIRISS Star Masking Library

This Python library is designed to mask bright stars in JWST NIRISS imaging mosaics. Foreground stars are identified from the Gaia catalog using the `astroquery` library. The NIRISS stellar PSF is approximated using a toy model, represented by a circular region for the PSF's central componenets, and extended rectangles for the diffraction spikes. The library also includes helper functions to check if a given source is contaminated by a stellar PSF.

The PSF is scaled for each object according to its Gaia G-band magnitude. The details of this calibration are included in `calibration.ipynb`.

## Functions
The library's key functions include:

### `get_footprint(mosaic, wcsdf, hdulistindex)`
Generates a footprint for each exposure in the mosaic.

- **Parameters:**
  - `mosaic` (THDUList): The mosaic FITS file.
  - `wcsdf` (pd.DataFrame): DataFrame containing WCS information.
  - `hdulistindex` (int): Index of the HDU in the mosaic.
- **Returns:**
  - `footprints` (list): List of footprints for each exposure.

### `mask_exposure(data, circradius, diffspikeradius, spikewidth, starxp, staryp, pa_of_star, ncores=None)`
Creates a mask for an exposure.

- **Parameters:**
  - `data` (ndarray): Image data.
  - `circradius` (iterable): Radius for masking each star.
  - `diffspikeradius` (iterable): Radii for diffraction spikes.
  - `spikewidth` (int): Width of the diffraction spikes.
  - `starxp` (ndarray): X-positions of stars.
  - `staryp` (ndarray): Y-positions of stars.
  - `pa_of_star` (float): Position angle of the exposure.
  - `ncores` (int, optional): Number of cores for multiprocessing.
- **Returns:**
  - `ndarray`: Mask representing PSFs of bright stars.

### `mask_mosaic(mosaic, wcspath, output_path=None, hdulistindex=0, spikewidth=30, ncores=None, inspect_final_mask=True, calibration_slope=-13.551, calibration_int=342.216)`
Creates a mask for an entire NIRISS mosaic composed of different exposures.

- **Parameters:**
  - `mosaic` (str or HDUList): Mosaic to mask.
  - `wcspath` (str): Path to WCS info file.
  - `output_path` (str): Path to save mosaic mask as FITS.
  - `hdulistindex` (int): Index in HDUList where data and header are located.
  - `spikewidth` (int): Width of diffraction spikes in NIRISS PSF mask.
  - `ncores` (int): Number of cores for multiprocessing.
  - `inspect_final_mask` (bool): If True, generates an image displaying the image and mask.
  - `calibration_slope` (float): Slope of calibration between PSF mask size and Gaia G-band magnitude.
  - `calibration_int` (float): Intercept of calibration between PSF mask size and Gaia G-band magnitude.
- **Returns:**
  - `HDUList`: Astropy HDUList object containing the star mask.

### `is_object_contaminated(starmaskhdu, objectdata)`
Checks if a given source is contaminated by a stellar PSF.

- **Parameters:**
  - `starmaskhdu` (HDUList): Star mask to check against.
  - `objectdata` (SkyCoord or HDUList or iterable): Objects to check if contaminated.
- **Returns:**
  - `bool or iterable of bool`: True/False indicating whether the object(s) is contaminated.

## Dependencies
The code has been tested in Python 3.12.2 with the following dependencies:
- `numpy`
- `pandas`
- `matplotlib`
- `astropy`
- `astroquery`
- `grizli`
- `pathos`
- `cv2`
- `tqdm`

## Installation

To install the required dependencies, you can use `pip`:

```bash
pip install numpy pandas matplotlib astropy astroquery grizli pathos opencv-python tqdm

