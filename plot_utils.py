import numpy as np
import pandas as pd
from pathlib import Path
from astropy.time import Time
import utils
from statsmodels.stats.weightstats import DescrStatsW


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


def get_lc_data(sn, band, sn_info):
    """
    Imports light curve file for specified SN and band. Cuts points with bad
    flags or sources outside detector radius, and also fixes duplicated headers.
    Inputs:
        sn (str): SN name
        band (str): 'FUV' or 'NUV'
        sn_info (DataFrame)
    Output:
        lc (DataFrame): light curve table
    """

    # Get name of light curve file
    fits_name = utils.sn2fits(sn, band)
    lc_file = LC_DIR / Path(fits_name.split('.')[0] + '.csv')
    # Discovery date
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')

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
    flags = [int(2 ** n) for n in range(0,10)]
    flag_count = [len(lc[lc['flags'] & f > 0]) for f in flags]
    fatal_flags = (1 | 2 | 4 | 16 | 64 | 128 | 512)
    lc = lc[lc['flags'] & fatal_flags == 0]

    # Cut sources outside detector radius
    plate_scale = 6 # as/pixel
    detrad_cut_px = DETRAD_CUT * 3600 / plate_scale
    lc = lc[lc['detrad'] < detrad_cut_px]

    # Cut ridiculous flux values
    lc = lc[np.abs(lc['flux_bgsub']) < 1]

    # Convert dates to MJD
    lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd
    lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd
    # Convert measured fluxes to absolute luminosities
    lc['luminosity'] = absolute_luminosity(sn, lc['flux_bgsub'], sn_info)
    lc['luminosity_err'] = absolute_luminosity(sn, lc['flux_bgsub_err'], sn_info)

    return lc, flag_count


def import_external_lc(sn, sn_info):
    try:
        lc = pd.read_csv(Path('external/%s_phot.csv' % sn))
        lc['magnitude_abs'], lc['e_magnitude_abs'] = absolute_mag(
                sn, lc['magnitude'], lc['e_magnitude'], sn_info)
        return lc
    except FileNotFoundError as e:
        raise e


def absolute_mag(sn, mags, mag_err, sn_info):
    """
    Converts apparent magnitudes to absolute magnitudes based on NED results
    Inputs:
        sn (str): SN name
        mags (Array-like): apparent magnitudes
        mag_err (Array-like): apparent magnitude
        sn_info (DataFrame): includes NED scrape results
    Outputs:
        absolute magnitudes (Array), absolute magnitude errors (Array)
    """

    h_dist = sn_info.loc[sn, 'h_dist'] # Mpc
    h_dist_err = sn_info.loc[sn, 'h_dist_err'] # Mpc
    mod = 5 * np.log10(h_dist * 1e6) - 5 # distance modulus
    mod_err = 5 * np.log10(h_dist_err * 1e6) - 5
    mod_errs = np.full(mod_err, mag_err.shape[0])
    return mags - mod, np.sqrt(mod_err ** 2 + mag_err ** 2)


def absolute_luminosity(sn, fluxes, sn_info):
    """
    Converts measured fluxes to absolute luminosities based on NED results
    Inputs:
        sn (str): SN name
        mags (Array-like): measured fluxes
        sn_info (DataFrame): includes NED scrape results
    Outputs:
        absolute luminosities (Array)
    """

    h_dist = sn_info.loc[sn, 'h_dist'] # Mpc
    h_dist_cm = h_dist * 3.08568e24 # cm
    luminosities = 4 * np.pi * h_dist_cm**2 * fluxes
    return luminosities