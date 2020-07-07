import numpy as np
import pandas as pd
from pathlib import Path
from astropy.time import Time
import utils
from statsmodels.stats.weightstats import DescrStatsW

LC_DIR = Path('/mnt/d/GALEXdata_v7/LCs/')
FITS_DIR = Path('/mnt/d/GALEXdata_v7/fits/')
EXTERNAL_LC_DIR = Path('external/')
DETRAD_CUT = 0.55 # deg
DT_MIN = -30
PLATE_SCALE = 6 # as/pixel


def absolute_luminosity(flux, dist, flux_err=None, dist_err=None):
    """
    Converts measured fluxes to absolute luminosities based on distance
    Inputs:
        flux (Array-like): measured fluxes
        flux_err (Array-like): measured flux error
        h_dist (float): distance in Mpc
        h_dist_err (float): distance error in Mpc
    Outputs:
        absolute luminosity (Array), luminosity error (Array)
    """

    cm_Mpc = 3.08568e24 # cm / Mpc
    luminosity = 4 * np.pi * (dist * cm_Mpc)**2 * flux
    if flux_err is None or dist_err is None:
        return luminosity
    else:
        err = np.abs(luminosity) * np.sqrt((2*dist_err/dist)**2 + (flux_err/flux)**2)
        return luminosity, err


def absolute_mag(mag, dist, mag_err_1=None, mag_err_2=None, dist_err=None):
    """
    Converts apparent magnitudes to absolute magnitudes based on distance
    Inputs:
        mags (Array-like): apparent magnitude(s)
        mag_err_1 (Array-like): apparent magnitude upper (brighter) error
        mag_err_2 (Array-like): apparent magnitude lower (dimmer) error
        h_dist (float): distance in Mpc
        h_dist_err (float): distance error in Mpc
    Outputs:
        absolute magnitude (Array), absolute magnitude errors (Arrays) if given
    """

    mod = 5 * np.log10(dist * 1e6) - 5 # distance modulus
    if mag_err_1 is None or dist_err is None:
        return mag - mod

    mod_err = np.abs(5 * dist_err / (dist * np.log(10)))
    if mag_err_2 is None:
        return mag - mod, np.sqrt(mod_err**2 + mag_err_1**2)

    return mag - mod, np.sqrt(mod_err**2 + mag_err_1**2), np.sqrt(mod_err**2 + mag_err_2**2)


def add_systematics(lc, bg, bg_err, sys_err, inplace=False):
    """
    Adds systematic errors in quadrature to flux and luminosity uncertainties;
    also subtracts host background from flux, luminosity, and magnitude
    Inputs:
        lc (DataFrame): light curve table
        bg, bg_err, sys_err (float): host background, background error, sys error
        inplace (bool, default False): whether to modify the original lc or return
            a new one
    Outputs:
        expanded light curve table
    """

    # Make copy of lc data frame to avoid modifying if inplace == False
    df = lc.copy()

    # Absolute luminosities of systematics
    # bg_lum, bg_err_lum = absolute_luminosity(bg, dist, bg_err, dist_err)
    # sys_err_lum = absolute_luminosity(sys_err, dist, dist_err=dist_err)

    # Add systematic error
    df['flux_bgsub_err_total'] = np.sqrt(df['flux_bgsub_err']**2 + sys_err**2)
    # lc['luminosity_err_total'] = np.sqrt(lc['luminosity_err']**2 + sys_err_lum**2)

    # Subtract host background
    df['flux_hostsub'] = df['flux_bgsub'] - bg
    df['flux_hostsub_err'] = np.sqrt(df['flux_bgsub_err_total']**2 + bg_err**2)
    # lc['luminosity_hostsub'] = lc['luminosity'] - bg_lum
    # lc['luminosity_hostsub_err'] = np.sqrt(lc['luminosity_err']**2 + bg_err**2)

    if inplace:
        lc = df
    else:
        return df


def get_background(lc):
    """
    Calculates the host background for a given light curve. Also calculates the
    systematic error needed to make the reduced chi squared value of the total
    error equal to 1.
    Inputs:
        lc (DataFrame): light curve table
    Outputs:
        bg (float): host background luminosity
        bg_err (float): host background luminosity error
        sys_err (float): systematic error based on reduced chi-squared test
    """

    before = lc[lc['t_delta'] < DT_MIN]
    data = np.array(before['flux_bgsub'])
    err = np.array(before['flux_bgsub_err'])
    # Need >1 point before discovery to add
    if len(before.index) > 1:
        # Initialize reduced chi-square, sys error values
        rcs = 2
        sys_err = 0
        sys_err_step = np.nanmean(err) * 0.1
        # Reduce RCS to 1 by adding systematic error in quadrature
        while rcs > 1:
            # Combine statistical and systematic error
            new_err = np.sqrt(err ** 2 + sys_err ** 2)
            # Determine background from weighted average of data before discovery
            weighted_stats = DescrStatsW(data, weights=1/new_err**2, ddof=0)
            bg = weighted_stats.mean
            bg_err = weighted_stats.std
            # Reduced chi squared test of data vs background
            rcs = utils.redchisquare(data, np.full(data.size, bg), new_err, n=0)
            # Increase systematic error for next iteration, if necessary
            sys_err += sys_err_step
    else:
        # TODO improve sys error estimate (from gPhoton)
        bg = lc.reset_index(drop=True).loc[0,'flux_bgsub']
        bg_err = lc.reset_index(drop=True).loc[0,'flux_bgsub_err']
        sys_err = 0.07 * bg

    return bg, bg_err, sys_err


def get_flags(sn, band):
    """
    Counts the number of each gPhoton flag present in the given light curve
    Inputs:
        sn (str): supernova name
        band (str): 'FUV' or 'NUV'
    Outputs:
        flag_count (list)
    """

    # Get name of light curve file
    fits_name = utils.sn2fits(sn, band)
    lc_file = LC_DIR / Path(fits_name.split('.')[0] + '.csv')
    # Read light curve data
    lc = pd.read_csv(lc_file)
    lc['flags'] = lc['flags'].astype(int)
    # Get flags
    flags = [int(2 ** n) for n in range(0,10)]
    flag_count = [len(lc[lc['flags'] & f > 0]) for f in flags]
    return flag_count


def import_lc(sn, band):
    """
    Imports light curve file for specified SN and band. Cuts points with bad
    flags or sources outside detector radius, and also fixes duplicated headers.
    Inputs:
        sn (str): SN name
        band (str): 'FUV' or 'NUV'
    Output:
        lc (DataFrame): light curve table
    """

    # Get name of light curve file
    fits_name = utils.sn2fits(sn, band)
    lc_file = LC_DIR / Path(fits_name.split('.')[0] + '.csv')

    # Read light curve data
    lc = pd.read_csv(lc_file)

    # Find duplicated headers, if any, and remove all duplicated material
    # then fix original file
    dup_header = lc[lc['t0'] == 't0']
    if len(dup_header) > 0:
        lc = lc.iloc[0:dup_header.index[0]]
        lc.to_csv(lc_file, index=False)
    lc = lc.astype(float)
    lc['flags'] = lc['flags'].astype(int)

    # Weed out bad flags
    fatal_flags = (1 | 2 | 4 | 16 | 64 | 128 | 512)
    lc = lc[lc['flags'] & fatal_flags == 0]

    # Cut sources outside detector radius
    detrad_cut_px = DETRAD_CUT * 3600 / PLATE_SCALE
    lc = lc[lc['detrad'] < detrad_cut_px]

    # Cut ridiculous flux values
    lc = lc[np.abs(lc['flux_bgsub']) < 1]

    # Convert dates to MJD
    lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd
    # disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
    # lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd

    # Convert measured fluxes to absolute luminosities
    # dist = sn_info.loc[sn, 'h_dist']
    # dist_err = sn_info.loc[sn, 'h_dist_err']
    # lc['luminosity'], lc['luminosity_err'] = absolute_luminosity(
    #         lc['flux_bgsub'], dist, lc['flux_bgsub_err'], dist_err)
    # lc['absolute_mag'], lc['absolute_mag_err_1'], lc['absolute_mag_err_2'] = \
    #         absolute_mag(lc['mag_bgsub'], dist, lc['mag_bgsub_err_1'],
    #                 lc['mag_bgsub_err_2'], dist_err)

    return lc


def import_external_lc(sn, sn_info):
    try:
        lc = pd.read_csv(EXTERNAL_LC_DIR / Path('%s_phot.csv' % sn))
        lc['magnitude_abs'], lc['e_magnitude_abs'] = absolute_mag(
                sn, lc['magnitude'], lc['e_magnitude'], sn_info)
        return lc
    except FileNotFoundError as e:
        raise e