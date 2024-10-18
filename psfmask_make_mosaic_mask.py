import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import WCS
import astropy.units as uu
from astroquery.gaia import Gaia
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from grizli.utils import log_scale_ds9
import sizecal
import cv2
from tqdm import tqdm
Gaia.ROW_LIMIT = int(1e6)

def patch_circle(array, center, radius, value):
    """Patches a circle onto a 2D NumPy array.

    Args:
        array (numpy.ndarray): The 2D array to modify.
        center (tuple): The (x, y) coordinates of the circle's center.
        radius (int): The radius of the circle.
        value (float): The value to fill the circle with.
    """
    x, y = np.ogrid[:array.shape[0], :array.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    array[mask] = value
    return array

def patch_rectangle(image, PA, height, width, xcen, ycen, value):
    mask = np.zeros(image.shape, dtype=np.uint8)
    theta = np.radians(90-PA)
    alpha = np.arctan(width/height)
    hyp = np.sqrt((width/2)*(width/2) + (height/2)*(height/2))
    x1 = xcen - hyp*np.cos(alpha+theta)
    y1 = ycen - hyp*np.sin(alpha+theta)
    x2, y2 = x1 + height*np.cos(theta), y1 + height*np.sin(theta)
    x3, y3 = x2 - width*np.sin(theta), y2 + width*np.cos(theta)
    x4, y4 = x1 - width*np.sin(theta), y1 + width*np.cos(theta)
    corners = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
    cv2.fillPoly(mask, [corners.astype(np.int32)], value)
    return mask

def get_star_pa(wcslist, stars):
    pa = np.zeros(len(stars))
    headers = [ww.to_header(relax=True) for ww in wcslist]
    centers=np.array([[hh['CRVAL1'],hh['CRVAL2']] for hh in headers])
    centers=SkyCoord(ra=centers[:,0]*uu.degree, dec=centers[:,1]*uu.degree, frame='icrs')
    for ii in range(0,len(stars)):
        for jj,wcs in enumerate(wcslist):
            if wcs.footprint_contains(SkyCoord(ra=stars['ra'][ii]*uu.degree, dec=stars['dec'][ii]*uu.degree, frame='icrs')):
                pa[ii] = np.arctan2(headers[jj]['PC2_1'], headers[jj]['PC2_2'])*180/np.pi
            else:
                # find which wcs is closest to the star and use its pa
                sep = SkyCoord(ra=stars['ra'][ii]*uu.degree, dec=stars['dec'][ii]*uu.degree, frame='icrs').separation(centers)
                closestidx = np.argmin(sep)
                pa[ii] = np.arctan2(headers[closestidx]['PC2_1'], headers[closestidx]['PC2_2'])*180/np.pi
    return -1*pa

def make_star_mask(mosaic, wcspath, output_path=None, hdulistindex=0, spikewidth=30, inspect_final_mask=True):
    """
    Create a mask for an entire mosaic or image.

    Inputs:
        output_path : str, path where mask should be saved to fits
        mosaic : filepath, astropy.fits.HDUList
        hdulistindex : int, index of relevant HDU in HDUList
        spikewidth : width in pixels of masks for diffraction spikes
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

    wcsdf = pd.read_csv(wcspath)
    wcs_exposures = [WCS(wcsdf.loc[ii].to_dict(), relax=True) for ii in wcsdf.index]
    pa_of_star = get_star_pa(wcs_exposures, stars_in_image)
    diffspikeradius = np.round(np.array([sizecal.scaledfit(gg) for gg in stars_in_image['phot_g_mean_mag']]), 0)
    circradius = np.round(0.1*diffspikeradius, 0)
    starxp, staryp = skycoord_to_pixel(SkyCoord(ra=stars_in_image['ra'], dec=stars_in_image['dec'], frame='icrs'), wcs=wcs)

    mask = np.zeros_like(data)
    angle_variations = [-2,-1,0,1,2] # deg
    primary_spike_angles = [0,60,120,180,240,300]
    secondary_spike_angles = [90,270]
    print("Masking diffraction spikes...")
    for ii, radius in tqdm(enumerate(circradius),total=len(circradius)):
        mask=patch_circle(mask, (staryp[ii], starxp[ii]), radius, 1)
        # make diffraction spikes
        for d_angle in angle_variations:
            for spike_angle in primary_spike_angles:
                mask+=patch_rectangle(mask, pa_of_star[ii]+d_angle+spike_angle, height=2*diffspikeradius[ii], width=spikewidth,
                        xcen=starxp[ii], ycen=staryp[ii], value=1)
            for spike_angle in secondary_spike_angles:
                mask+=patch_rectangle(mask, pa_of_star[ii]+d_angle+spike_angle, height=0.5*diffspikeradius[ii], width=spikewidth,\
                        xcen=starxp[ii], ycen=staryp[ii], value=1)
    mask = (mask>0).astype(int)
    if inspect_final_mask:
        fig,axs=plt.subplots(ncols=3,figsize=(11,3))
        axs[0].imshow(mask)
        axs[0].set_title("Star Mask")
        axs[1].imshow(log_scale_ds9(data))
        axs[1].set_title("Original Data")
        axs[2].imshow(log_scale_ds9((1-mask)*data))
        axs[2].set_title("Masked Data")
        plt.tight_layout()
        plt.savefig(output_path.replace('fits','png'),dpi=300)
        plt.show()
    if output_path is not None:
        fits.HDUList(fits.PrimaryHDU(data=mask, header=hdulist[hdulistindex].header)).writeto(output_path, overwrite=True)
    return mask

if __name__=='__main__':
    import time
    ti=time.time()
    root = '/Volumes/T7/data/mpia/mosaics/'
    #_=make_star_mask(root+'aqr-01-ir_drc_sci.fits', root+'aqr-01-f200wn-clear_wcs.csv', 'aqr-01.fits', inspect_final_mask=True)
    #_=make_star_mask(root+'boo-06-ir_drc_sci.fits', root+'boo-06-f200wn-clear_wcs.csv', 'boo-06.fits', inspect_final_mask=True)
    #_=make_star_mask(root+'boo-05-ir_drc_sci.fits', root+'boo-05-f200wn-clear_wcs.csv', 'boo-05.fits', inspect_final_mask=True)
    #_=make_star_mask(root+'vir-12-ir_drc_sci.fits', root+'vir-12-f200wn-clear_wcs.csv', 'vir-12.fits', inspect_final_mask=True)
    #_=make_star_mask(root+'uma-03-ir_drc_sci.fits', root+'uma-03-f200wn-clear_wcs.csv', 'uma-03.fits', inspect_final_mask=True)
    #_=make_star_mask(root+'sex-09-ir_drc_sci.fits', root+'sex-09-f200wn-clear_wcs.csv', 'sex-09.fits', inspect_final_mask=True)

    print('done in {:0.2f} min'.format((time.time()-ti)/60))
