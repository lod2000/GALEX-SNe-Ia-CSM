#!/usr/bin/env python

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import argparse

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import Angle

from photutils import CircularAperture
from photutils import aperture_photometry
from photutils import CircularAnnulus
from photutils.utils import calc_total_error

import utils

'''
Multi-aperture photometry using a circular aperture with sigma-clipped annulus
local background subtraction.

    data: single FITS image array (2D)
    expt: exposure time
    band: 'NUV' or 'FUV' (affects error calculation)
    positions: list of tuples - locations of aperture center;
        should be (float, float) for pixels or (Angle, Angle) for SkyCoord
    wcs: World Coordinate System from fits header
    r: aperture radius
    r_in: inner annulus radius
    r_out: outer annulus radius
'''
def multi_aper_phot(data, expt, band, positions, wcs, r=3., r_in=6., r_out=9.):
    # convert image from counts to counts per second (cps)
    data = data / expt
    # convert sky positions (ra, dec) to pixels using wcs
    for i, pos in enumerate(positions):
        if type(pos[0]).__name__ == 'Angle':
            sky_pos = SkyCoord(pos[0], pos[1])
            positions[i] = sky_pos.to_pixel(wcs)

    # set up circular & annulus apertures
    aperture = CircularAperture(positions, r) # inner aperture
    annulus_aperture = CircularAnnulus(positions, r_in, r_out)
    apers = [aperture, annulus_aperture]
    annulus_masks = annulus_aperture.to_mask(method='center')

    # error estimation
    # detector background: 10^-4 cps/pixel FUV or 10^-3 cps/pixel NUV
    effective_gain = expt # exposure time in seconds
    if band=='FUV': bkg_error = 0.0001 * data
    else: bkg_error = 0.001 * data
    error = calc_total_error(data, bkg_error, effective_gain)

    # sigma-clipped background median
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)

    # photometry
    phot = aperture_photometry(data, apers, error=error)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * aperture.area
    phot['aper_bkg_err'] = phot['aperture_sum_err_1']
    # aperture sum - sigma clipped median background
    phot['aper_sum_bkgsub'] = phot['aperture_sum_0'] - phot['aper_bkg']
    phot['aper_sum_bkgsub_err'] = np.sqrt(phot['aper_bkg_err'] ** 2 + \
            phot['aperture_sum_err_0'] ** 2)

    return phot


'''
Run photometry on all images in a single fits file

fits_file: pathlib.Path
positions: list of positions (pixels or SkyCoords)
'''
def fits_phot(fits_file, positions, r=3., r_in=6., r_out=9.):
    fits = utils.Fits(fits_file)

    sn = []
    img = []
    cps = []
    cps_err = []
    if len(fits.expts) > 0: # Only check if there's exposure data
        for i, img in enumerate(fits.data):
            phot = multi_aper_phot(img, fits.expts[i], fits.band, positions,
                    fits.wcs, r, r_in, r_out)
            cps.append(phot['aper_sum_bkgsub'])
            cps_err.append(phot['aper_sum_bkgsub_err'])
        sn = [fits.sn.name] * fits.data.shape[0]
        img = range(fits.data.shape[0])
        cps = np.array(cps)
        cps_err = np.array(cps_err)
        '''
        # Find images (if any) with significantly higher photometry than average,
        # AND which are after the discovery date of the supernova
        indices = [[sn, sn.band, i] for i in range(len(data)) if 
            sums[i] > np.mean(sums) + 2 * np.std(sums) 
            and tmeans[i] > sn.disc_date.gps]
        return indices
        '''
    return [fits.filename, cps, cps_err]


'''
Run photometry on center of all images in a single fits file

fits_file: pathlib.Path
'''
def sn_phot(fits_file, r=3., r_in=6., r_out=9.):
    f = utils.Fits(fits_file)
    if len(f.expts) > 0:
        positions = [(f.header['CRPIX1'], f.header['CRPIX2'])]
        if f.header['NAXIS'] == 2:
            f.data = np.array([f.data])

        baseline_phot = multi_aper_phot(f.data[0], f.expts[0], f.band, positions,
                f.wcs, r, r_in, r_out)
        baseline_cps = baseline_phot['aper_sum_bkgsub']
        baseline_err = baseline_phot['aper_sum_bkgsub_err']

        cps = []
        cps_err = []
        for i, img in enumerate(f.data):
            phot = multi_aper_phot(img, f.expts[i], f.band, positions,
                    f.wcs, r, r_in, r_out)
            cps.append(phot['aper_sum_bkgsub'] - baseline_cps)
            cps_err.append(np.sqrt(phot['aper_sum_bkgsub_err']**2 + baseline_err**2))
        filename = np.array([[f.filename]] * f.data.shape[0])
        img = np.array([np.arange(f.data.shape[0])]).transpose()
        cps = np.array(cps)
        cps_err = np.array(cps_err)
        '''
        # Find images (if any) with significantly higher photometry than average,
        # AND which are after the discovery date of the supernova
        indices = [[sn, sn.band, i] for i in range(len(data)) if 
            sums[i] > np.mean(sums) + 2 * np.std(sums) 
            and tmeans[i] > sn.disc_date.gps]
        return indices
        '''
        return np.hstack([filename, img, cps, cps_err])
    else:
        return np.array([[f.filename, np.nan, np.nan, np.nan]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Returns photometry at center\
            of each FITS image.')
    parser.add_argument('fits_dir', metavar='dir', type=Path, 
            help='path to FITS data directory')
    args = parser.parse_args()

    # Read clean OSC csv
    osc = utils.import_osc(Path('ref/OSC-pre2014-v2-clean.csv'))

    fits_dir = args.fits_dir

    # Read classification csv, selecting those with images pre and post discovery.
    # or multiple images post-discovery
    classified = pd.read_csv(Path('out/fits_categories.csv'), index_col='File')
    fits_files = classified[(classified['Category'] == 'post_disc') | \
            (classified['Category'] == 'pre_post_disc')].index
    fits_files = [fits_dir / f for f in fits_files]
    #fits_files = [f for f in fits_dir.glob('**/*.fits.gz')]

    with mp.Pool() as pool:
        cps = list(tqdm(pool.imap(sn_phot, fits_files), total=len(fits_files)))

    df = pd.DataFrame(np.concatenate(cps), columns=['File', 'Epoch', 'CPS', 'CPS Error'])
    #df = df[np.logical_not(pd.isna(df['CPS']))]
    try:
        df.to_csv('out/cps.csv', index=False)
    except PermissionError:
        df.to_csv('out/cps-tmp.csv', index=False)

    '''
    for fits_file in tqdm(utils.get_fits_files(
            Path('/mnt/exthdd/GALEX_SNeIa_REU/fits/'), 'pre_post_discovery.csv')
            ):
        indices = single_fits_phot(fits_file)
        if len(indices)>0:
            noteworthy.append(indices)

    noteworthy = np.concatenate(np.array(noteworthy))
    np.savetxt('noteworthy.csv', noteworthy, delimiter=',', fmt='%s')
    '''

    #fits_path = Path('/mnt/d/GALEX_SNeIa_REU/fits/CSS100912_223118+010516-NUV.fits.gz')
    #print(sn_aperture_phot(fits_path))
