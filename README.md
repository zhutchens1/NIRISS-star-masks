# Bright Star Masks for NIRISS JWST Imaging

Construct bright star masks for JWST NIRISS imaging based on the Gaia catalog (astroquery search).

- `sourcefinding_make_mosaic_mask.py`: constructs a star mask by empirically identifying stellar PSFs in imaging data
- `psfmask_make_mosaic_mask.py`: constructs a star mask by placing toy-model NIRISS PSFs at known star locations
- `sizecal.py`: supplement file for `sourcefinding_make_mosaic_mask.py` that calibrates approximate diffraction spike sizes vs. Gaia G-band mag
