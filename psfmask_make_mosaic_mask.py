import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import NDData
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import WCS, FITSFixedWarning
import astropy.units as uu
from astroquery.gaia import Gaia
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from grizli.utils import log_scale_ds9
from scipy.ndimage import binary_fill_holes
from pathos.multiprocessing import ProcessingPool 
import sizecal
import cv2
import math
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FITSFixedWarning)
Gaia.ROW_LIMIT = int(1e6)

def get_footprint(mosaic,wcsdf,hdulistindex):
    output_projection = WCS(mosaic[hdulistindex].header)
    shape_out = mosaic[hdulistindex].data.shape
    footprints=[]
    for ii in range(0,len(wcsdf.index)):
        wcs = WCS(wcsdf.loc[ii,:].to_dict())
        im=NDData(data=np.full((wcsdf.loc[ii,'naxis1'],wcsdf.loc[ii,'naxis2']), fill_value=1),wcs=wcs)
        fp = np.zeros_like(mosaic[hdulistindex].data)
        XX,YY = np.meshgrid(np.arange(wcsdf.loc[ii,'naxis1']), np.arange(wcsdf.loc[ii,'naxis2']))
        ra,dec = wcs.wcs_pix2world(XX, YY, 0)
        coords = SkyCoord(ra=ra*uu.degree,dec=dec*uu.degree,frame='icrs')
        #idx = coords.contained_by(wcs)
        #fp[np.where(idx)] = 1
        idx_in_mosaic = output_projection.world_to_array_index(coords)
        fp[idx_in_mosaic]=1
        fp = fill_footprint_gaps(fp)
        footprints.append(fp)
    return footprints

def fill_footprint_gaps(image):
    for ii in range(0,image.shape[0]):
        zerosel = np.where(image[ii,:]==0)[0]
        onesel = np.where(image[ii,:]==1)[0]
        if len(onesel)>0:
            min1 = np.min(onesel)
            max1 = np.max(onesel)
            zerofillsel = zerosel[np.where(np.logical_and(zerosel>min1, zerosel<max1))]
            image[ii,zerofillsel] = 1
    return image

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
    theta = (90-PA)*math.pi/180.
    alpha = math.atan(width/height)
    hyp = math.sqrt((width/2)*(width/2) + (height/2)*(height/2))
    x1 = xcen - hyp*math.cos(alpha+theta)
    y1 = ycen - hyp*math.sin(alpha+theta)
    x2, y2 = x1 + height*math.cos(theta), y1 + height*math.sin(theta)
    x3, y3 = x2 - width*math.sin(theta), y2 + width*math.cos(theta)
    x4, y4 = x1 - width*math.sin(theta), y1 + width*math.cos(theta)
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

def mask_exposure(data, circradius, diffspikeradius, spikewidth, starxp, staryp, pa_of_star, ncores=None):
    """
    data - image data as np.array
    circradius - iterable, contains radius for which to mask each star
    starxp - xpositions of stars
    staryp - ypositions of stars
    pa_of_star (scalar) - PA of exposure

    returns: mask (1=star 0=no star)
    """
    mask = np.zeros_like(data)
    angle_variations = [-2,-1,0,1,2] # deg
    primary_spike_angles = [0,60,120,180,240,300]
    secondary_spike_angles = [90,270]
    if ncores is None:
        for ii, radius in enumerate(circradius):
            mask+=mask_individual_star(ii, mask, circradius, starxp, staryp, pa_of_star, angle_variations, primary_spike_angles, secondary_spike_angles, spikewidth, diffspikeradius)       
    else:
        def worker_fn(ii):
            return mask_individual_star(ii,mask, circradius, starxp, staryp, pa_of_star, angle_variations, primary_spike_angles, secondary_spike_angles, spikewidth, diffspikeradius)
        pool=ProcessingPool(ncores)
        masks = pool.map(worker_fn, np.arange(circradius.shape[0]))
        mask = np.sum(masks,axis=0)
    return mask

def mask_individual_star(ii, mask, circradius, starxp, staryp, pa_of_star, angle_variations, primary_spike_angles, secondary_spike_angles, spikewidth, diffspikeradius):
    mask=patch_circle(mask, (staryp[ii], starxp[ii]), circradius[ii], 1)
    for d_angle in angle_variations:
        for spike_angle in primary_spike_angles:
            mask+=patch_rectangle(mask, pa_of_star+d_angle+spike_angle, height=2*diffspikeradius[ii], width=spikewidth,
                    xcen=starxp[ii], ycen=staryp[ii], value=1)
        for spike_angle in secondary_spike_angles:
            mask+=patch_rectangle(mask, pa_of_star+d_angle+spike_angle, height=0.5*diffspikeradius[ii], width=spikewidth,\
                    xcen=starxp[ii], ycen=staryp[ii], value=1)
    return mask

def mask_mosaic(mosaic, wcspath, output_path=None, hdulistindex=0, spikewidth=30, ncores=None, inspect_final_mask=True):
    """
    Create a mask for an entire mosaic or image.

    Inputs:
        output_path : str, path where mask should be saved to fits
        mosaic : filepath, astropy.fits.HDUList
        hdulistindex : int, index of relevant HDU in HDUList
        spikewidth : width in pixels of masks for diffraction spikes
        ncores : number of cores to use for multiprocessing (pathos)
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

    wcsdf.loc[:,'pa'] = np.round(-1*np.arctan2(wcsdf['cd2_1'],wcsdf['cd2_2'])*180/np.pi) # nearest degree good enough.
    fps = np.array(get_footprint(hdulist, wcsdf, hdulistindex))

    uniquepas = np.unique(wcsdf.pa.to_list())
    masks = []
    for pa in tqdm(uniquepas,total=len(uniquepas)):
        masks.append(mask_exposure(data,circradius,diffspikeradius,spikewidth,starxp,staryp,pa,ncores=ncores))
    masks = np.array(masks)
    masks = np.array([masks[np.where(uniquepas==pa)][0] for pa in wcsdf.pa.to_list()])
    mask = np.sum(masks * fps, axis=0)
    mask = (mask>0).astype(int)
    mask = fits.HDUList(fits.PrimaryHDU(data=mask, header=hdulist[hdulistindex].header))
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
        mask.writeto(output_path, overwrite=True)
    return mask

def is_object_contaminated(starmaskhdu, objectdata):
    """
    Check if a given source is contaminated by a stellar PSF.

    Parameters
    ---------------
    starmaskhdu : astropy.io.fits.HDUList object
        Star mask to check against. (1 = star, 0 = no star)
    objectdata : astropy.coordinates.SkyCoord object, astropy.io.fits.HDUList object, or iterable
        Objects to check if contaminated. If the input is a SkyCoord, the function assesses whether
        the star mask's value at the object's RA/Dec is 1 or 0. If passing an image, it should be a
        segmentation map, in which case the function will check for any overlap between the source
        and the star mask.
        If iterable, elements must be SkyCoord or HDUList. 

    Returns
    --------------
    contaminated : bool or iterable of bool
        True/False indicating whether the object(s) is contaminated.
    """
    try:
        iter(objectdata)
    except TypeError:
        objectdata = [objectdata]
    if isinstance(objectdata[0],fits.HDUList):
        contaminated=_is_object_contaminated_hdulist(starmaskhdu, objectdata)
    elif isinstance(objectdata[0],SkyCoord):
        contaminated=_is_object_contaminated_skycoord(starmaskhdu, objectdata)
    else:
        raise TypeError("`objectdata` must be (iterable of) astropy SkyCoord or HDUList, not type %s" % type(objectdata[0]))
    return (contaminated if (len(contaminated)>1) else contaminated[0])

def _is_object_contaminated_skycoord(starmaskhdu, objectdata):
    contaminated = np.zeros(len(objectdata)).astype(bool)
    maskwcs = WCS(starmaskhdu[0].header)
    for ii,object_ in enumerate(objectdata):
        contaminated[ii] = bool(starmaskhdu[0].data[maskwcs.world_to_array_index(object_)])
    return contaminated

def _is_object_contaminated_hdulist(starmaskhdu, objectdata):
    contaminated = np.zeros(len(objectdata)).astype(bool)
    maskwcs = WCS(starmaskhdu[0].header)
    for ii,object_ in enumerate(objectdata):
        objhdr = object_[0].header
        XX,YY = np.meshgrid(np.arange(objhdr.get('naxis1')), np.arange(objhdr.get('naxis2')))
        ra,dec = wcs.wcs_pix2world(XX, YY, 0)
        objposition = SkyCoord(ra=np.mean(ra), dec=np.mean(dec), frame='icrs')
        maskcutout = Cutout2D(starmaskhdu[0].data, position=objposition, size=object_[0].data.shape, wcs=maskwcs)
        contaminated[ii] = bool(np.sum(object_[0].data * maskcutout[0].data) > 0)
    return contaminated

#######################################################
#######################################################
#######################################################
#######################################################

if __name__=='__main__':
    import time
    ti=time.time()
    ncpu = 55
    root = './'#/Volumes/T7/data/mpia/mosaics/'
    #_=mask_mosaic(root+'aqr-01-ir_drc_sci.fits', root+'aqr-01-f200wn-clear_wcs.csv', 'aqr-01.fits', inspect_final_mask=True)
    #_=mask_mosaic(root+'boo-06-ir_drc_sci.fits', root+'boo-06-f200wn-clear_wcs.csv', 'boo-06.fits', inspect_final_mask=True)
    #_=mask_mosaic(root+'boo-05-ir_drc_sci.fits', root+'boo-05-f200wn-clear_wcs.csv', 'boo-05.fits', inspect_final_mask=True, ncores=ncpu)
    #_=mask_mosaic(root+'vir-12-ir_drc_sci.fits', root+'vir-12-f200wn-clear_wcs.csv', 'vir-12.fits', inspect_final_mask=True)
    _=mask_mosaic(root+'uma-03-ir_drc_sci.fits', root+'uma-03-f200wn-clear_wcs.csv', 'uma-03.fits', inspect_final_mask=False, ncores=ncpu)
    #_=mask_mosaic(root+'sex-09-ir_drc_sci.fits', root+'sex-09-f200wn-clear_wcs.csv', 'sex-09.fits', inspect_final_mask=True)

    print(is_object_contaminated(_,SkyCoord(ra=189.4192819*uu.degree, dec=62.2029112*uu.degree, frame='icrs')))
    print(is_object_contaminated(_,SkyCoord(ra=189.3848905*uu.degree, dec=62.2253592*uu.degree, frame='icrs')))
    print(is_object_contaminated(_,SkyCoord(ra=0*uu.degree,dec=0*uu.degree,frame='icrs')))

    print('done in {:0.2f} min'.format((time.time()-ti)/60))
