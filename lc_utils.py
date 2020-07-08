import numpy as np
import pandas as pd
from pathlib import Path
from astropy.time import Time
import utils
from statsmodels.stats.weightstats import DescrStatsW

LC_DIR = Path('/mnt/d/GALEXdata_v9/LCs/')
FITS_DIR = Path('/mnt/d/GALEXdata_v9/fits/')
EXTERNAL_LC_DIR = Path('external/')
DETRAD_CUT = 0.55 # deg
DT_MIN = -30
PLATE_SCALE = 6 # as/pixel


def absolute_luminosity(flux, dist):
    """
    Converts measured fluxes to absolute luminosities based on distance
    Inputs:
        flux (Array-like): measured fluxes
        dist (float): distance in Mpc
    Outputs:
        absolute luminosity (Array)
    """

    cm_Mpc = 3.08568e24 # cm / Mpc
    luminosity = 4 * np.pi * (dist * cm_Mpc)**2 * flux
    return luminosity


def absolute_luminosity_err(flux, flux_err, dist, dist_err):
    """
    Converts measured fluxes to absolute luminosities based on distance, and
    also returns corresponding error
    Inputs:
        flux (Array-like): measured fluxes
        flux_err (Array-like): measured flux error
        dist (float): distance in Mpc
        dist_err (float): distance error in Mpc
    Outputs:
        absolute luminosity (Array), luminosity error (Array)
    """

    luminosity = absolute_luminosity(flux, dist)
    err = np.abs(luminosity) * np.sqrt((2*dist_err/dist)**2 + (flux_err/flux)**2)
    return luminosity, err


def absolute_mag(mag, dist):
    """
    Converts apparent magnitudes to absolute magnitudes based on distance
    Inputs:
        mag (Array-like): apparent magnitude(s)
        dist (float): distance in Mpc
    Outputs:
        absolute magnitude (Array)
    """

    mod = 5 * np.log10(dist * 1e6) - 5 # distance modulus
    return mag - mod


def absolute_mag_err(mag, mag_err, dist, dist_err):
    """
    Converts apparent magnitudes to absolute magnitudes based on distance
    Inputs:
        mag (Array-like): apparent magnitude(s)
        mag_err (Array-like): apparent magnitude error
        dist (float): distance in Mpc
        dist_err (float): distance error in Mpc
    Outputs:
        absolute magnitude (Array), absolute magnitude error (Array)
    """

    mod = 5 * np.log10(dist * 1e6) - 5 # distance modulus
    mod_err = np.abs(5 * dist_err / (dist * np.log(10)))
    return mag - mod, np.sqrt(mod_err**2 + mag_err**2)


def add_systematics(lc, quantity='all'):
    """
    Adds systematic errors in quadrature to flux and luminosity uncertainties;
    also subtracts host background from flux, luminosity, and magnitude
    Inputs:
        lc (DataFrame): light curve table
        bg, bg_err, sys_err (float): host background, background error, sys error
    Outputs:
        expanded light curve table
    """

    if quantity in ('flux', 'all'):
        # Get background & systematic error
        bg, bg_err, sys_err = get_background(lc, 'flux')
        # Add systematic error
        lc['flux_bgsub_err_total'] = np.sqrt(lc['flux_bgsub_err']**2 + sys_err**2)
        # Subtract host background
        lc['flux_hostsub'] = lc['flux_bgsub'] - bg
        lc['flux_hostsub_err'] = np.sqrt(lc['flux_bgsub_err_total']**2 + bg_err**2)

    if quantity in ('luminosity', 'all'):
        # Get background & systematic error
        bg, bg_err, sys_err = get_background(lc, 'luminosity')
        # Add systematic error
        lc['luminosity_err_total'] = np.sqrt(lc['luminosity_err']**2 + sys_err**2)
        # Subtract host background
        lc['luminosity_hostsub'] = lc['luminosity'] - bg
        lc['luminosity_hostsub_err'] = np.sqrt(lc['luminosity_err']**2 + bg_err**2)

    return lc


def get_background(lc, quantity='flux'):
    """
    Calculates the host background for a given light curve. Also calculates the
    systematic error needed to make the reduced chi squared value of the total
    error equal to 1.
    Inputs:
        lc (DataFrame): light curve table
        quantity (str): 'flux' or 'luminosity' or 'magnitude' (WIP), default 'flux'
    Outputs:
        bg (float): host background luminosity
        bg_err (float): host background luminosity error
        sys_err (float): systematic error based on reduced chi-squared test
    """

    if quantity == 'flux':
        header = 'flux_bgsub' 
    elif quantity == 'luminosity':
        header = 'luminosity'

    before = lc[lc['t_delta'] < DT_MIN]
    data = np.array(before[header])
    err = np.array(before['%s_err' % header])
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
        bg = lc.reset_index(drop=True).loc[0, header]
        bg_err = lc.reset_index(drop=True).loc[0, '%s_err' % header]
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

    return lc


def import_external_lc(sn, dist, dist_err):
    """
    Imports light curve file from an external source (e.g. Swift)
    """

    lc = pd.read_csv(EXTERNAL_LC_DIR / Path('%s_phot.csv' % sn))
    lc['absolute_mag'], lc['absolute_mag_err'] = absolute_mag(lc['magnitude'], 
            dist, mag_err_1=lc['e_magnitude'], dist_err=dist_err)
    lc['']
    return lc


def improve_lc(lc, sn, sn_info):

    # Convert dates to MJD
    lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd

    # Add days relative to discovery date
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
    lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd

    # Convert measured fluxes to absolute luminosities
    dist = sn_info.loc[sn, 'h_dist']
    dist_err = sn_info.loc[sn, 'h_dist_err']
    lc['luminosity'], lc['luminosity_err'] = absolute_luminosity_err(
            lc['flux_bgsub'], lc['flux_bgsub_err'], dist, dist_err)

    # Convert apparent to absolute magnitudes
    lc['absolute_mag'], lc['absolute_mag_err_1'] = absolute_mag_err(
            lc['mag_bgsub'], lc['mag_bgsub_err_1'], dist, dist_err)
    lc['absolute_mag_err_2'] = absolute_mag_err(
            lc['mag_bgsub'], lc['mag_bgsub_err_2'], dist, dist_err)[1]

    return lc


def swift_vega2ab(vega_mag, vega_mag_err, filter):
    """
    Converts Vega magnitudes from Swift to AB magnitudes, based on the filter
    Inputs:
        vega_mag (float or Array): Swift Vega magnitude
        vega_mag_err (float or Array): Swift Vega magnitude error
        filter (str): uvw1, uvm2, or uvw2
    """
    
    baseline = {'uvw1': 1.51, 'uvm2': 1.69, 'uvw2': 1.73}
    c = baseline[filter]
    return vega_mag + c, np.sqrt(vega_mag_err**2 + 0.03**2)