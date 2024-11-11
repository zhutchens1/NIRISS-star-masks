import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import NDData
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import WCS, FITSFixedWarning
from astropy.nddata import Cutout2D
import astropy.units as uu
from astroquery.gaia import Gaia
from grizli.utils import log_scale_ds9
from pathos.multiprocessing import ProcessingPool 
import cv2
import math
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FITSFixedWarning)
Gaia.ROW_LIMIT = int(1e6)

def get_footprint(mosaic,wcsdf,hdulistindex):
    """
    Generate a footprint for each exposure in the mosaic.

    Parameters
    --------------------
    mosaic : HDUList
        The mosaic FITS file.
    wcsdf : pd.DataFrame
        DataFrame containing WCS information.
    hdulistindex : int
        Index of the HDU in the mosaic.

    Returns
    --------------------
    footprints : list
        List of footprints for each exposure. Length
        matches len(wcsdf).
    """
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
    """
    Fill gaps in the footprint image.

    Parameters
    --------------------
    image : ndarray
        The footprint image.

    Returns
    --------------------
    ndarray : The footprint image with gaps filled.
    """
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
    """
    Patch a circle onto a 2D NumPy array.

    Parameters
    --------------------
    array : ndarray
        The 2D array to modify.
    center : tuple
        The (x, y) coordinates of the circle's center.
    radius : int
        The radius of the circle.
    value : float
        The value to fill the circle with.

    Returns
    --------------------
    ndarray: The modified array with the circle patched.
    """
    x, y = np.ogrid[:array.shape[0], :array.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    array[mask] = value
    return array

def patch_rectangle(image, PA, height, width, xcen, ycen, value):
    """
    Patch a rectangle onto a 2D image.

    Parameters
    --------------------
    image : ndarray
        The 2D array to modify.
    PA : float
        Position angle, in degrees, of the rectangle.
    height : int
        Height of the rectangle.
    width : int
        Width of the rectangle.
    xcen : int
        X-coordinate of the rectangle's center.
    ycen : int
        Y-coordinate of the rectangle's center.
    value : float
        The value to fill the rectangle with.

    Returns
    --------------------
    ndarray: The modified array with the rectangle patched.
    """
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

def get_exposure_pa(wcslist, stars):
    """
    Get the position angles of exposures.

    Parameters
    --------------------
    wcslist : list
        List of WCS objects.
    stars : pd.DataFrame
        DataFrame containing star information.

    Returns
    --------------------
    ndarray
        Array of position angles for each exposure, length
        matches wcslist.
    """
    pa = np.zeros(len(stars))
    headers = [ww.to_header(relax=True) for ww in wcslist]
    centers=np.array([[hh['CRVAL1'],hh['CRVAL2']] for hh in headers])
    centers=SkyCoord(ra=centers[:,0]*uu.degree, dec=centers[:,1]*uu.degree, frame='icrs')
    for ii in range(0,len(stars)):
        for jj,wcs in enumerate(wcslist):
            if wcs.footprint_contains(SkyCoord(ra=stars['ra'][ii]*uu.degree, dec=stars['dec'][ii]*uu.degree, frame='icrs')):
                pa[ii] = np.arctan2(headers[jj]['PC2_1'], headers[jj]['PC2_2'])*180/np.pi
            else:
                sep = SkyCoord(ra=stars['ra'][ii]*uu.degree, dec=stars['dec'][ii]*uu.degree, frame='icrs').separation(centers)
                closestidx = np.argmin(sep)
                pa[ii] = np.arctan2(headers[closestidx]['PC2_1'], headers[closestidx]['PC2_2'])*180/np.pi
    return -1*pa

def mask_subexposure_within_mosaic(data, circradius, diffspikeradius, spikewidth, starxp, staryp, pa_of_star, ncores=None):
    """
    Create a mask for an exposure.

    Parameters
    --------------------
    data : ndarray
        Image data.
    circradius : iterable
        Radius for masking each star.
    diffspikeradius : iterable
        Radii for diffraction spikes.
    spikewidth : int
        Width of the diffraction spikes.
    starxp : ndarray
        X-positions of stars.
    staryp : ndarray
        Y-positions of stars.
    pa_of_star : float
        Position angle of the exposure.
    ncores : int, optional
        Number of cores for multiprocessing.
        If None (default), multiprocessing 
        is not used.

    Returns
    --------------------
    ndarray: Mask representing PSFs of bright stars.
    """
    mask = np.zeros_like(data)
    angle_variations = [-2,-1,0,1,2] # deg
    primary_spike_angles = [0,60,120,180,240,300]
    secondary_spike_angles = [90,270]
    if ncores is None:
        for ii, radius in enumerate(circradius):
            mask+=mask_individual_star(ii, mask, circradius, starxp, staryp, pa_of_star, angle_variations, primary_spike_angles, \
                    secondary_spike_angles, spikewidth, diffspikeradius)       
    else:
        def worker_fn(ii):
            return mask_individual_star(ii,mask, circradius, starxp, staryp, pa_of_star, angle_variations, primary_spike_angles, \
                    secondary_spike_angles, spikewidth, diffspikeradius)
        pool=ProcessingPool(ncores)
        masks = pool.map(worker_fn, np.arange(circradius.shape[0]))
        mask = np.sum(masks,axis=0)
    return mask

def mask_individual_star(ii, mask, circradius, starxp, staryp, pa_of_exposure, angle_variations, primary_spike_angles, secondary_spike_angles, spikewidth, diffspikeradius):
    """
    Mask a star at a given position and PA using the NIRISS PSF, as set by user-specified size.
    This function is written for a context in which many stars need to be masked, hence the index
    `ii` and iterable arguments (e.g. starxp representing x positions for *all* stars) as inputs.
    To mask a single star in a single image, use ii=0 and make all arguments length-1 iterables,
    e.g. circradius = [5] rather than 5.

    All angles should be supplied in degrees, and lengths/radii in units of pixels.

    Parameters
    --------------------
    ii : int
        index of star in array.
    mask : ndarray
        Image mask to modify.
    circradius : iterable
        radius of central PSF component (circle) to mask.
    starxp : iterable
        x position of star.
    staryp : iterable
        y position of star.
    pa_of_exposure : scalar
        PA of exposure.
    angle_variations : iterable
        variations on the PA to mask, to account for 
        errors in centering/PA (e.g. [-1,0,1] degrees).
    primary_spike_angles : iterable
        angles of primary NIRISS PSF spikes.
    secondary_spike_angles : iterables
        angles of secondary NIRISS PSF spikes.
    spikewidth : scalar
        width of diffraction spikes within mask, pixels.
    diffspikeradius : iterable
        radial length of diffraction spike for the given star `ii`. Length matches `circradius`.

    Returns
    --------------------
    ndarray : Modified mask with star at index `ii` masked.
    """
    mask=patch_circle(mask, (staryp[ii], starxp[ii]), circradius[ii], 1)
    for d_angle in angle_variations:
        for spike_angle in primary_spike_angles:
            mask+=patch_rectangle(mask, pa_of_exposure+d_angle+spike_angle, height=2*diffspikeradius[ii], width=spikewidth,
                    xcen=starxp[ii], ycen=staryp[ii], value=1)
        for spike_angle in secondary_spike_angles:
            mask+=patch_rectangle(mask, pa_of_exposure+d_angle+spike_angle, height=0.75*diffspikeradius[ii], width=spikewidth,\
                    xcen=starxp[ii], ycen=staryp[ii], value=1)
    return mask

def mask_mosaic(mosaic, wcspath, output_path=None, hdulistindex=0, spikewidth=30, ncores=None, inspect_final_mask=True,\
    calibration_slope = -13.551, calibration_int = 342.216):
    """
    Create a mask for an entire NIRISS mosaic composed of different exposures.

    Parameters
    ------------------
    mosaic : str or HDUList
        mosaic to mask. If str, it should represent a file path to a FITS image.
    wcspath : str
        path to WCS info file (WCS csv file generated by grizli).
    output_path : str
        path to save mosaic mask as FITS.
    hdulistindex : int
        Index in HDUList (`mosaic`) where data and header are located. Default 0.
    spikewidth : int
        Width of diffraction spikes in NIRISS PSF mask, in pixels. Default 30.
    ncores : int
        Number of cores for use in multiprocessing. If None, multiprocessing is
        not used.
    inspect_final_mask : bool
        If True, the function will generate an image displaying the image and mask.
    calibration_slope : float
        Slope of calibration between PSF mask size and Gaia G-band magnitude.
        Units = pixels/mag, default value derived in `calibration.ipynb`.
    calibration_int : float
        Intercept of calibration between PSF mask size and Gaia G-band magnitude.
        Units = pixels, default value derived in `calibration.ipynb`.

    Returns
    ------------------
    HDUList : astropy HDUList object containing the star mask. Matches the 
        WCS of the original mosaic.
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
    pa_of_star = get_exposure_pa(wcs_exposures, stars_in_image)
    calibration = lambda gmag: calibration_slope*gmag + calibration_int
    circradius = np.round(np.array([calibration(gg) for gg in stars_in_image['phot_g_mean_mag']]), 0)
    diffspikeradius = (10*circradius).astype(int)
    starxp, staryp = skycoord_to_pixel(SkyCoord(ra=stars_in_image['ra'], dec=stars_in_image['dec'], frame='icrs'), wcs=wcs)

    wcsdf.loc[:,'pa'] = np.round(-1*np.arctan2(wcsdf['cd2_1'],wcsdf['cd2_2'])*180/np.pi) # nearest degree good enough.
    fps = np.array(get_footprint(hdulist, wcsdf, hdulistindex))

    uniquepas = np.unique(wcsdf.pa.to_list())
    masks = []
    for pa in tqdm(uniquepas,total=len(uniquepas)):
        masks.append(mask_subexposure_within_mosaic(data,circradius,diffspikeradius,spikewidth,starxp,staryp,pa,ncores=ncores))
    masks = np.array(masks)
    masks = np.array([masks[np.where(uniquepas==pa)][0] for pa in wcsdf.pa.to_list()])
    mask = np.sum(masks * fps, axis=0)
    mask = (mask>0).astype(int)
    mask = fits.HDUList(fits.PrimaryHDU(data=mask, header=hdulist[hdulistindex].header))
    if inspect_final_mask:
        fig,axs=plt.subplots(ncols=3,figsize=(11,3))
        axs[0].imshow(mask[0].data)
        axs[0].set_title("Star Mask")
        axs[1].imshow(log_scale_ds9(data))
        axs[1].set_title("Original Data")
        axs[2].imshow(log_scale_ds9((1-mask[0].data)*data))
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
    starmaskhdu : HDUList object
        Star mask to check against. (1 = star, 0 = no star)
    objectdata : astropy.coordinates.SkyCoord object, astropy.io.fits.ImageHDU or PrimaryHDU object, or iterable
        Objects to check if contaminated. If the input is a SkyCoord, the function assesses whether
        the star mask's value at the object's RA/Dec is 1 or 0. If passing an image, it should be a
        segmentation map, in which case the function will check for any overlap between the source
        and the star mask.
        If iterable, elements must be SkyCoord or HDUList.
    extension : int or str
        FITS extension in objectdata elements. Only used if passing image data, not SkyCoord.
        Default 0.

    Returns
    --------------
    contaminated : bool or iterable of bool
        True/False indicating whether the object(s) is contaminated.
    """
    try:
        iter(objectdata)
    except TypeError:
        objectdata = [objectdata]
    if isinstance(objectdata[0],fits.PrimaryHDU) or isinstance(objectdata[0],fits.ImageHDU):
        contaminated=_is_object_contaminated_hdulist(starmaskhdu, objectdata)
    elif isinstance(objectdata[0],SkyCoord):
        contaminated=_is_object_contaminated_skycoord(starmaskhdu, objectdata)
    else:
        raise TypeError("`objectdata` must be (iterable of) astropy SkyCoord or PrimaryHDU, not type %s" % type(objectdata[0]))
    return (contaminated if (len(contaminated)>1) else contaminated[0])

def _is_object_contaminated_skycoord(starmaskhdu, objectdata):
    """
    Check if a given source, described by a SkyCoord, is contaminated
    by a stellar PSF. See `is_object_contaminated` for details.
    """
    contaminated = np.zeros(len(objectdata)).astype(bool)
    maskwcs = WCS(starmaskhdu[0].header)
    for ii,object_ in enumerate(objectdata):
        if maskwcs.footprint_contains(object_):
            contaminated[ii] = bool(starmaskhdu[0].data[maskwcs.world_to_array_index(object_)])
        else:
            contaminated[ii] = False
    return contaminated

def _is_object_contaminated_hdulist(starmaskhdu, objectdata):
    """
    Check if a given source, described by an HDUList, is contaminated
    by a stellar PSF. See `is_object_contaminated` for details.
    """
    contaminated = np.zeros(len(objectdata)).astype(bool)
    maskwcs = WCS(starmaskhdu[0].header)
    for ii,object_ in enumerate(objectdata):
        print(object_)
        objhdr = object_.header
        XX,YY = np.meshgrid(np.arange(objhdr.get('naxis1')), np.arange(objhdr.get('naxis2')))
        ra,dec = WCS(objhdr).wcs_pix2world(XX, YY, 0)
        objposition = SkyCoord(ra=np.mean(ra)*uu.degree, dec=np.mean(dec)*uu.degree, frame='icrs')
        maskcutout = Cutout2D(starmaskhdu[0].data, position=objposition, size=object_.data.shape, wcs=maskwcs)
        contaminated[ii] = bool(np.sum(object_.data * maskcutout.data) > 0)
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

    #print(is_object_contaminated(_,SkyCoord(ra=189.4192819*uu.degree, dec=62.2029112*uu.degree, frame='icrs')))
    #print(is_object_contaminated(_,SkyCoord(ra=189.3848905*uu.degree, dec=62.2253592*uu.degree, frame='icrs')))
    #print(is_object_contaminated(_,SkyCoord(ra=0*uu.degree,dec=0*uu.degree,frame='icrs')))

    sourcehdu = fits.open('../data_outthere/dr0.1/all_source_data/data.outthere-survey.org/0.1/objects/fits/outthere-hudfn_36334.beams.fits')
    print(sourcehdu['REF'])
    print(is_object_contaminated(_,sourcehdu['REF']))

    print('done in {:0.2f} min'.format((time.time()-ti)/60))
