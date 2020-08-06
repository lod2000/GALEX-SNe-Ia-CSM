from pathlib import Path
import platform

from operator import or_
from functools import reduce

import pandas as pd
import numpy as np

LC_DIR = Path('/mnt/d/GALEXdata_v10/LCs/')
PLATE_SCALE = 6 # (as/pixel)

class LightCurve:
    """GALEX light curve of a given supernova and band."""
    def __init__(self, fname, dir=LC_DIR, detrad_cut=0.55, fatal_flags=[], 
            manual_cuts=[]):
        """Import a light curve file.

        Arguments:
            fname: file name, should be similar to 'SN2007on-FUV.csv'
            dir: parent directory for light curve files
            detrad_cut: maximum detector radius value in degrees
            fatal_flags: additional fatal flags to eliminate, list of ints
            manual_cuts: indices to remove 'manually', list of ints
        """

        self.fname = Path(fname)
        self.sn, self.band = fname2sn(fname)

        # Import light curve CSV
        data = pd.read_csv(Path(dir) / self.fname, dtype={'flags': int})

        # Remove fatal flags
        self.fatal_flags = [1, 2, 4, 16, 64, 128, 512] + fatal_flags
        data = data[data['flags'] & reduce(or_, self.fatal_flags) == 0]

        # Cut sources outside detector radius
        detrad_cut_px = detrad_cut * 3600 / PLATE_SCALE
        data = data[data['detrad'] < detrad_cut_px]

        # Cut data with background much higher than average to eliminate washed-
        # out fields
        bg_cps = data['bg_counts'] / data['exptime']
        bg_median = np.median(bg_cps)
        high_bg = bg_cps[bg_cps > 3 * bg_median]
        data = data[~data.index.isin(high_bg.index)]

        # Cut unphysical data
        data = data[np.abs(data['flux_bgsub']) < 1] # extreme flux values
        data = data[data['bg_counts'] >= 0] # negative background counts

        # Manual cuts
        data = data[~data.index.isin(manual_cuts)]

        # Cleaned-up data
        self.data = data

        # Host background


def get_background(lc, band):
    """
    Calculates the host background for a given light curve. Also calculates the
    systematic error needed to make the reduced chi squared value of the total
    error equal to 1. In cases with only a handful of points before discovery,
    the systematic error is approximated by the gAperture photometric reliability
    fit (see gAper_sys_err).
    Inputs:
        lc (DataFrame): light curve table
        band (str): 'FUV' or 'NUV'
    Outputs:
        bg (float): host background
        bg_err (float): host background error; includes systematic error
        sys_err (float): systematic error based on reduced chi-squared test
    """

    before = lc[lc['t_delta'] < DT_MIN]
    data = np.array(before['flux_bgsub'])
    err = np.array(before['flux_bgsub_err'])

    # For many background points, calculate the systematic error using a
    # reduced chi-squared fit
    if len(before.index) > 4:
        # Initialize reduced chi-square, sys error values
        rcs = 2
        sys_err_step = np.nanmean(err) * 0.1
        sys_err = -sys_err_step
        # Reduce RCS to 1 by adding systematic error in quadrature
        while rcs > 1:
            # Increase systematic error for next iteration
            sys_err += sys_err_step
            # Combine statistical and systematic error
            new_err = np.sqrt(err ** 2 + sys_err ** 2)
            # Determine background from weighted average of data before discovery
            weighted_stats = DescrStatsW(data, weights=1/new_err**2, ddof=0)
            bg = weighted_stats.mean
            bg_err = np.sqrt(weighted_stats.std**2 + sys_err**2)
            # Reduced chi squared test of data vs background
            rcs = redchisquare(data, np.full(data.size, bg), new_err, n=0)

    # For few background points, use the polynomial fit of |MCAT - gAper| errors
    # based on the original gAperture (non-background-subtracted) magnitudes
    elif len(before.index) > 1:
        # Determine background from weighted average of data before discovery
        weighted_stats = DescrStatsW(data, weights=1/err**2, ddof=0)
        bg = weighted_stats.mean
        bg_err = weighted_stats.std
        # Use background if it's positive, or annulus flux if it's not
        if bg > 0:
            bg_mag = galex_flux2mag(bg, band)
            # Calculate systematic error from gAperture photometric error 
            sys_err_mag = gAper_sys_err(bg_mag, band)
            # Convert mag error to SNR
            snr = 1 / (10 ** (sys_err_mag / 2.5) - 1)
            sys_err = bg / snr
        else:
            ann_flux = np.average(before['flux'] - before['flux_bgsub'], weights=1/err**2)
            bg_mag = galex_flux2mag(ann_flux, band)
            # Calculate systematic error from gAperture photometric error 
            sys_err_mag = gAper_sys_err(bg_mag, band)
            # Convert mag error to SNR
            snr = 1 / (10 ** (sys_err_mag / 2.5) - 1)
            sys_err = ann_flux / snr
        # Include systematic in background uncertainty
        bg_err = np.sqrt(bg_err ** 2 + sys_err ** 2)

    # Otherwise, just use the first point
    else:
        bg = lc['flux_bgsub'].iloc[0]
        bg_err = lc['flux_bgsub_err'].iloc[0]
        # Use background if it's positive, or annulus flux if it's not
        if bg > 0:
            bg_mag = galex_flux2mag(bg, band)
            # Calculate systematic error from gAperture photometric error 
            sys_err_mag = gAper_sys_err(bg_mag, band)
            # Convert mag error to SNR
            snr = 1 / (10 ** (sys_err_mag / 2.5) - 1)
            sys_err = bg / snr
        else:
            ann_flux = lc['flux'].iloc[0] - lc['flux_bgsub'].iloc[0]
            bg_mag = galex_flux2mag(ann_flux, band)
            # Calculate systematic error from gAperture photometric error 
            sys_err_mag = gAper_sys_err(bg_mag, band)
            # Convert mag error to SNR
            snr = 1 / (10 ** (sys_err_mag / 2.5) - 1)
            sys_err = ann_flux / snr
        # Include systematic in background uncertainty
        bg_err = np.sqrt(bg_err ** 2 + sys_err ** 2)

    return bg, bg_err, sys_err


def fname2sn(fname):
    """Extract SN name and band from a file name."""

    fname = Path(fname)
    split = fname.stem.split('-')
    sn = '-'.join(split[:-1])
    band = split[-1]
    # Windows replaces : with _ in some file names
    if 'CSS' in sn or 'MLS' in sn:
        sn.replace('_', ':', 1)
    sn.replace('_', ' ')
    return sn, band


def sn2fname(sn, band, suffix='.csv'):
    # Converts SN name and GALEX band to a file name, e.g. for a light curve CSV

    fname = '-'.join((sn, band)) + suffix
    fname.replace(' ', '_')
    # Make Windows-friendly
    if (platform.system() == 'Windows') or ('Microsoft' in platform.release()):
        fname = fname.replace(':', '_')
    return Path(fname)

