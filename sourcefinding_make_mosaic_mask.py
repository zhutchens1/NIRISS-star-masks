import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from photutils.aperture import CircularAperture
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources
from astropy.stats import SigmaClip
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.nddata import NDData
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as uu
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from astroquery.gaia import Gaia
from scipy.integrate import simpson
from scipy.stats import mode
from scipy.ndimage import binary_fill_holes
Gaia.ROW_LIMIT = int(1e6)

from grizli.utils import log_scale_ds9

def get_star_segm_number(segm, visual_inspect):
    """
    finds the source ID number of the star in the segmentation map 
    
    Inputs:
        segm : 2D np.array representing segmentation array
        visual_inspect : bool. If True, user will be prompted for visual 
                         inspection wherever the star cannot be auto-identified.
    Outputs:
        sid : int, source ID of star in segmentation array
        star_in_center : bool, denotes whether the segm map shows the star
                         at the center of the segmentation array. Sometimes
                         the saturated stars are missing the central part of
                         the PSF and thus are not foudn at the center of the
                         seg array.
    """
    # try to first get pixel value at very center
    sid = segm[segm.shape[0]//2, segm.shape[1]//2]
    star_in_center=True
    if (sid==0) or np.isnan(sid):
        star_in_center=False
        # get mode of non-zero/non-NaN segm values in middle-third image
        cutout = Cutout2D(segm, position=(segm.shape[0]//2, segm.shape[1]//2), size=int(segm.shape[0]/10.))
        fl = cutout.data.flatten()
        sid,_ = mode(fl[fl!=0], nan_policy='omit')
        if (sid==0) or np.isnan(sid):
            all_nan_in_middle = np.isnan(fl.all()) or (fl==0).all()
            if not (all_nan_in_middle):
                if visual_inspect and (not all_nan_in_middle):
                    print('Identify the segmentation ID of the star in the center of the image: ')
                    plt.figure()
                    plt.imshow(segm)
                    plt.show()
                    user_decision = input('Enter a number (the source ID) or `n` if there is none.')
                    sid = (None if (user_decision=='n') else user_decision)
                else:
                    print('********** cant find a star ID automatically! Consider turning on visual identification.')
                    sid = None
            else:
                sid = None
    return sid, star_in_center

def make_star_mask(mosaic, output_path=None, hdulistindex=0, cutout_box_size=3000, threshold_nsigma=1.5, visual_inspect=False, inspect_final_mask=True):
    """
    Create a mask for an entire mosaic or image.

    Inputs:
        output_path : str, path where mask should be saved to fits
        mosaic : filepath, astropy.fits.HDUList
        hdulistindex : int, index of relevant HDU in HDUList
	cutout_box_size : int, width of square cutout around each Gaia star (3000 is good for HUDF-N)
	threshold_nsigma : float, number of standard deviations above noise for source detection
	visual_inspect : bool, allows user to manually identify star in ambiguous cases. This is strongly recommended.
	inspect_final_mask : if True, the code displays the final mask for visual inspection.
    Outputs:
        mask : 1/0 mask representing the locations of bright stars

    """
    if isinstance(mosaic, str):
        hdulist = fits.open(mosaic)
    else:
        hdulist = mosaic
    data = hdulist[hdulistindex].data
    wcs = WCS(hdulist[hdulistindex].header)
    (bottomleft, topleft, topright, bottomright) = wcs.calc_footprint()
    bottomleft = SkyCoord(ra=bottomleft[0], dec=bottomleft[1], unit=(uu.degree, uu.degree), frame=hdulist[hdulistindex].header['RADESYS'].lower())
    topright = SkyCoord(ra=topright[0], dec=topright[1], unit=(uu.degree, uu.degree), frame=hdulist[hdulistindex].header['RADESYS'].lower())
    searchradius = bottomleft.separation(topright)
    stars = Gaia.query_object(coordinate=bottomleft, radius=searchradius)
    starscoord = SkyCoord(ra=stars['ra'], dec=stars['dec'], frame='icrs')
    in_image_idx = wcs.footprint_contains(starscoord)
    stars_in_image = stars[in_image_idx]
    stars_in_image = stars_in_image[(stars_in_image['classprob_dsc_combmod_star']>0.5)]
    
    masks=[]
    for ii in range(0,len(stars_in_image)):
        cutout_ii = Cutout2D(data = data, wcs = wcs, position = SkyCoord(stars_in_image['ra'][ii]*uu.degree, stars_in_image['dec'][ii]*uu.degree,\
                frame='icrs'), size = cutout_box_size)
        if ((cutout_ii.data==0).all() or (np.isnan(cutout_ii.data).all())):
            print("Passing on star index %d -- cutout contains no data" % ii)
        else:
            sigma_clip = SigmaClip(sigma=3.0,maxiters=10)
            thresh = detect_threshold(cutout_ii.data, nsigma=threshold_nsigma, sigma_clip=sigma_clip)
            segm = detect_sources(cutout_ii.data, thresh, npixels=10)
            segm = deblend_sources(cutout_ii.data, segm, npixels=100, contrast=0.4) # 0.001 
            star_segm_number, in_center = get_star_segm_number(segm.data, visual_inspect)
            segm.remove_labels([lb for lb in segm.labels if ((lb!=star_segm_number) and (lb!=0))],relabel=True)
            if in_center:
                mask_ii = segm.data # =1 where the star is
            else:
                mask_ii = binary_fill_holes(segm.data)
            masks.append(NDData(data=mask_ii,wcs=cutout_ii.wcs))
    finalmask,fp = reproject_and_coadd(masks, output_projection=wcs, shape_out=data.shape, reproject_function=reproject_interp)
    finalmask[finalmask>0]=1

    if inspect_final_mask:
        fig,axs=plt.subplots(ncols=3)
        axs[0].imshow(finalmask)
        axs[0].set_title("Star Mask")
        axs[1].imshow(log_scale_ds9(data))
        axs[1].set_title("Original Data")
        axs[2].imshow(log_scale_ds9((1-finalmask)*data))
        axs[2].set_title("Masked Data")
        plt.tight_layout()
        plt.show()
    if output_path is not None:
        fits.HDUList(fits.PrimaryHDU(data=finalmask, header=hdulist[hdulistindex].header)).writeto(output_path, overwrite=True)
    return finalmask
            

if __name__=='__main__':
    import time
    ti=time.time()
    _=make_star_mask('/Volumes/T7/outthere-hudfn-f200wn-clear_drc_sci.fits', './test.fits', visual_inspect=True)
    print('done in {:0.2f} min'.format((time.time()-ti)/60))
    
