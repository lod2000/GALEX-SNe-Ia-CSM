import pandas as pd
import numpy as np

from pathlib import Path
import platform

from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS
from statsmodels.stats.weightstats import DescrStatsW

# Default file and directory paths
LC_DIR = Path('/mnt/d/GALEXdata_v10/LCs/')      # light curve data dir
FITS_DIR = Path('/mnt/d/GALEXdata_v10/fits/')   # FITS data dir
OSC_FILE = Path('ref/osc.csv')                  # Open Supernova Catalog file
EXTERNAL_LC_DIR = Path('external/')             # light curves from Swift / others
BG_FILE = Path('out/high_bg.csv')               # discarded data with high background
EMPTY_LC_FILE = Path('out/empty_lc.csv')        # SNe with no useful lc data

# Variable cut parameters
DETRAD_CUT = 0.55   # Detector radius above which to cut (deg)
DT_MIN = -30        # Separation between background and SN data (days)

# GALEX spacecraft info
PLATE_SCALE = 6 # (as/pixel)
LAMBDA_EFF = {'FUV': 1549, 'NUV': 2304.7} # angstroms

# Physical constants
C = 3e8 # meter/second

# Plot color palette
COLORS = {'FUV' : '#a37', 'NUV' : '#47a', # GALEX
          'UVW1': '#cb4', 'UVM2': '#283', 'UVW2': '#6ce', # Swift
          'F275W': '#e67', # Hubble
          'g': 'c', 'r': 'r', 'i': 'y', 'z': 'brown', 'y': 'k' # Pan-STARRS
          }


################################################################################
## General utilities
################################################################################

def output_csv(df, file, **kwargs):
    """
    Outputs pandas DataFrame to CSV. Since Excel doesn't allow file modification
    while it's open, this function will write to a temporary file instead, to 
    ensure that the output of a long script isn't lost.
    Inputs:
        df (DataFrame): DataFrame to write
        file (str or Path): file name to write to
        **kwargs: passed on to DataFrame.to_csv()
    """

    if type(file) == str:
        file = Path(file) 
    try:
        df.to_csv(file, **kwargs)
    except PermissionError:
        tmp_file = file.parent / Path(file.stem + '-tmp' + file.suffix)
        df.to_csv(tmp_file, **kwargs)


# Reduced chi squared statistic
def redchisquare(data, model, sd, n=0):
    chisq = np.sum(((data-model)/sd)**2)
    return chisq/(len(data)-1-n)


def gAper_sys_err(mag, band):
    """
    Calculates the systematic error from gAperture at a given magnitude
    based on Michael's polynomial fits
    """

    coeffs = {
            'FUV': [4.07675572e-04, -1.98866713e-02, 3.24293442e-01, -1.75098239e+00],
            'NUV': [3.38514034e-05, -2.88685479e-03, 9.88349458e-02, -1.69681516e+00,
                    1.45956431e+01, -5.02610071e+01]
    }
    fit = np.poly1d(coeffs[band])
    return fit(mag)


################################################################################
## Conversions
################################################################################

"""
All conversion functions are designed to accept either float or NumPy array-like
inputs, as long as all input arrays have the same shape. The 'band' should be a
string or array of strings, matching one of the keys of the below conversion
factors.
"""

# https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
GALEX_ZERO_POINT    = {'FUV': 18.82, 'NUV': 20.08}
GALEX_FLUX_FACTOR   = {'FUV': 1.4e-15, 'NUV': 2.06e-16}
# Bianchi values for Milky Way extinction (over E(B-V)), from Table 2
GALEX_EXTINCTION    = {'FUV': 8.06, 'NUV': 7.95, 'U': 4.72, 'B': 4.02, 'V': 3.08}
# Poole et al. 2008 (per Angstrom)
SWIFT_ZERO_POINT    = {'V': 17.89, 'B': 19.11, 'U': 18.34, 'UVW1': 17.49, 
                       'UVM2': 16.82, 'UVW2': 17.35}
SWIFT_ZERO_ERROR    = {'V': 0.013, 'B': 0.016, 'U': 0.020, 'UVW1': 0.03, 
                       'UVM2': 0.03, 'UVW2': 0.03}
SWIFT_FLUX_FACTOR   = {'V': 2.614e-16, 'B': 1.472e-16, 'U': 1.63e-16, 
                       'UVW1': 4.3e-16, 'UVM2': 7.5e-16, 'UVW2': 6.0e-16}
SWIFT_FLUX_ERROR    = {'V': 8.7e-19, 'B': 5.7e-19, 'U': 2.5e-18,
                       'UVW1': 2.1e-17, 'UVM2': 1.1e-16, 'UVW2': 6.4e-17}
# Breeveld et al. 2011
SWIFT_AB_CONVERSION = {'V':-0.01, 'B':-0.13, 'U': 1.02, 'UVW1': 1.51, 
                       'UVM2': 1.69, 'UVW2': 1.73}
SWIFT_AB_CONV_ERROR = {'V': 0.01, 'B': 0.02, 'U': 0.02, 'UVW1': 0.03, 
                       'UVM2': 0.03, 'UVW2': 0.03}


def freq2wavelength(flux, wavelength):
    # Converts flux density from per Hz to per wavelength (A)
    # Input wavelength: effective wavelength of filter, in angstroms
    return flux * C * 1e10 / wavelength**2


def wavelength2freq(flux, wavelength):
    # Converts flux density from per Angstrom to per Hz
    # Input wavelength: effective wavelength of filter, in angstroms
    return flux * wavelength**2 / (C * 1e10)


def galex_cps2flux(cps, band):
    # Converts GALEX CPS to flux values
    return np.vectorize(GALEX_FLUX_FACTOR.get)(band) * cps


def galex_cps2mag(cps, band):
    # Converts GALEX CPS to AB magnitudes
    return -2.5 * np.log10(cps) + np.vectorize(GALEX_ZERO_POINT.get)(band)


def galex_extinction(a_v, band):
    # Converts foreground extinction A_V to FUV or NUV extinction in mags
    E = a_v / GALEX_EXTINCTION['V'] # E(B-V)
    a_band = E * GALEX_EXTINCTION[band] # A_FUV or A_NUV
    return a_band # magnitudes


def galex_flux2mag(flux, band):
    # Converts fluxes from GALEX to AB magnitudes
    cps = flux / np.vectorize(GALEX_FLUX_FACTOR.get)(band)
    return galex_cps2mag(cps, band)


def galex_mag2cps_err(mag, mag_err, band):
    # Converts AB magnitudes measured by GALEX into flux
    cps = 10 ** (2/5 * (np.vectorize(GALEX_ZERO_POINT.get)(band) - mag))
    cps_err = cps * (2/5) * np.log(10) * mag_err
    return cps, cps_err


def galex_delta_mag(cps, band, exp_time):
    # Estimates photometric repeatability vs magnitudes based on GALEX counts
    factor = 0.05 if band=='FUV' else 0.027
    return -2.5 * (np.log10(cps) - np.log10(cps + np.sqrt(cps * exp_time + \
            (factor * cps * exp_time) ** 2) / exp_time))


def swift_vega2ab(vega_mag, vega_mag_err, band):
    # Converts Vega magnitudes from Swift to AB magnitudes
    conv = np.vectorize(SWIFT_AB_CONVERSION.get)(band)
    conv_err = np.vectorize(SWIFT_AB_CONV_ERROR.get)(band)
    ab_mag_err = np.sqrt(vega_mag_err**2 + conv_err**2)
    return vega_mag + conv, ab_mag_err


def swift_mag2cps(mag, mag_err, band):
    # Converts Swift AB magnitudes to CPS
    diff = np.vectorize(SWIFT_ZERO_POINT.get)(band) - mag
    diff_err = np.sqrt(mag_err**2 + np.vectorize(SWIFT_ZERO_ERROR.get)(band)**2)
    cps = 10 ** (2/5 * diff)
    cps_err = cps * (2/5) * np.log(10) * diff_err
    return cps, cps_err


def swift_cps2flux(cps, cps_err, band):
    # Converts Swift CPS to flux values
    c = np.vectorize(SWIFT_FLUX_FACTOR.get)(band)
    c_err = np.vectorize(SWIFT_FLUX_ERROR.get)(band)
    flux = cps * c
    flux_err = flux * np.sqrt((cps_err/cps)**2 + (c_err/c)**2)
    return flux, flux_err


def flux2luminosity(flux, dist, z, a_v, band):
    """
    Converts measured fluxes to absolute luminosities based on distance,
    accounting for redshift and extinction
    Inputs:
        flux (Array-like): measured fluxes
        dist (float): distance in Mpc
        z (float): redshift
        a_v (float): V band extinction in magnitudes
    Outputs:
        absolute luminosity (Array)
    """

    cm_Mpc = 3.08568e24 # cm / Mpc
    a_band = galex_extinction(a_v, band)
    # Calculate total luminosity
    luminosity = 4 * np.pi * (dist * cm_Mpc)**2 * flux
    # Correct for redshift
    luminosity *= (1 + z) ** 3
    # Correct for extinction
    luminosity *= 10 ** (0.4 * a_band)
    return luminosity


def absolute_luminosity_err(flux, flux_err, dist, dist_err, z, z_err, a_v, band):
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

    luminosity = flux2luminosity(flux, dist, z, a_v, band)
    z_corr_err = np.nan_to_num(3 * z_err / (1+z))
    err = np.abs(luminosity) * np.sqrt((2*dist_err/dist)**2 + (flux_err/flux)**2 + z_corr_err**2)
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


################################################################################
## Light curve data
################################################################################

def check_if_empty(lc, sn, band):
    """
    Checks if a light curve DataFrame is empty after all the cuts during
    import_lc. If it is, append it to a file and raise an error.
    """

    if len(lc.index) == 0:
        empty_lc = pd.DataFrame([[sn, band]], columns=['name', 'band'])
        if EMPTY_LC_FILE.is_file():
            empty_lc = pd.read_csv(EMPTY_LC_FILE).append(empty_lc)
            empty_lc.drop_duplicates(inplace=True)
        output_csv(empty_lc, EMPTY_LC_FILE, index=False)
        raise KeyError


def full_import(sn, band, sn_info):
    """
    Imports the light curve for a specified supernova and band, adds luminosity
    and days since discovery from SN info file, and incorporates background
    and systematic errors
    """

    lc = import_lc(sn, band)

    # Convert dates to MJD
    lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd

    # Add days relative to discovery date
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
    lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd

    # Correct epoch for stretch factor
    lc['t_delta_rest'] = 1 / (1 + sn_info.loc[sn, 'z']) * lc['t_delta']

    # Get background & systematic error
    bg, bg_err, sys_err = get_background(lc, band)
    # Add systematic error
    lc['flux_bgsub_err_total'] = np.sqrt(lc['flux_bgsub_err']**2 + sys_err**2)
    # Subtract host background
    lc['flux_hostsub'] = lc['flux_bgsub'] - bg
    lc['flux_hostsub_err'] = np.sqrt(lc['flux_bgsub_err']**2 + bg_err**2)

    # Convert measured fluxes to absolute luminosities
    dist = sn_info.loc[sn, 'pref_dist']
    dist_err = sn_info.loc[sn, 'pref_dist_err']
    z = sn_info.loc[sn, 'z']
    z_err = sn_info.loc[sn, 'z_err']
    a_v = sn_info.loc[sn, 'a_v']
    lc['luminosity'], lc['luminosity_err'] = absolute_luminosity_err(
            lc['flux_bgsub'], lc['flux_bgsub_err_total'], dist, dist_err, z, 
            z_err, a_v, band)
    lc['luminosity_hostsub'], lc['luminosity_hostsub_err'] = absolute_luminosity_err(
            lc['flux_hostsub'], lc['flux_hostsub_err'], dist, dist_err, z,
            z_err, a_v, band)
    
    # Flux & luminosity density in terms of Hz
    convert_cols = ['flux_bgsub', 'flux_bgsub_err', 'flux_bgsub_err_total',
            'flux_hostsub', 'flux_hostsub_err', 'luminosity', 'luminosity_err',
            'luminosity_hostsub', 'luminosity_hostsub_err']
    for col in convert_cols:
        hz_col = col+'_hz'
        lc[hz_col] = wavelength2freq(lc[col], LAMBDA_EFF[band])

    # Convert apparent to absolute magnitudes
    lc['absolute_mag'], lc['absolute_mag_err_1'] = absolute_mag_err(
            lc['mag_bgsub'], lc['mag_bgsub_err_1'], dist, dist_err)
    lc['absolute_mag_err_2'] = absolute_mag_err(
            lc['mag_bgsub'], lc['mag_bgsub_err_2'], dist, dist_err)[1]

    return lc, bg, bg_err, sys_err


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
    fits_name = sn2fits(sn, band)
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
    fits_name = sn2fits(sn, band)
    lc_file = LC_DIR / Path(fits_name.split('.')[0] + '.csv')

    # Read light curve data
    lc = pd.read_csv(lc_file)

    # Find duplicated headers, if any, and remove all duplicated material
    # then fix original file
    if 't0' in lc['t0']:
        dup_header = lc[lc['t0'] == 't0']
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

    # Cut data with background counts less than 0
    lc = lc[lc['bg_counts'] >= 0]

    check_if_empty(lc, sn, band)

    # Cut data with background much higher than average (washed-out fields)
    # and output high backgrounds to file
    lc.insert(29, 'bg_cps', lc['bg_counts'] / lc['exptime'])
    bg_median = np.median(lc['bg_cps'])
    high_bg = lc[lc['bg_cps'] > 3 * bg_median]
    if len(high_bg.index) > 0:
        high_bg.insert(30, 'bg_cps_median', [bg_median] * len(high_bg.index))
        high_bg.insert(0, 'name', [sn] * len(high_bg.index))
        high_bg.insert(1, 'band', [band] * len(high_bg.index))
        if BG_FILE.is_file():
            high_bg = pd.read_csv(BG_FILE, index_col=0).append(high_bg)
            high_bg.drop_duplicates(inplace=True)
        output_csv(high_bg, BG_FILE, index=True)
        lc = lc[lc['bg_counts'] < 3 * bg_median]

    # Add manual cuts (e.g. previously identified as a ghost image)
    manual_cuts = pd.read_csv(Path('ref/manual_cuts.csv'))
    to_remove = manual_cuts[(manual_cuts['name'] == sn) & (manual_cuts['band'] == band)]['index']
    lc = lc[~lc.index.isin(to_remove)]

    check_if_empty(lc, sn, band)
    # Add dummy row if lc is otherwise empty
    # if len(lc.index) == 0:
    #     raise
    #     lc.loc[0,:] = np.full(len(lc.columns), np.nan)
        # print('%s has no valid data points in %s!' % (sn, band))

    return lc


def import_panstarrs(sn, sn_info):
    """
    Imports CSV of detections from Pan-STARRS catalog search
    """
    # Assuming order is g, r, i, z, y; from Tonry et al. 2012
    filters = {1: 4866, 2: 6215, 3: 7545, 4: 8679, 5: 9633} # angstrom

    lc = pd.read_csv(EXTERNAL_LC_DIR / Path('%s_panstarrs.csv' % sn))

    # Add days relative to discovery date
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
    lc['t_delta'] = lc['obsTime'] - disc_date.mjd

    # Sky subtraction
    aperture_area = np.pi * lc['apRadius'] ** 2
    lc['apFluxSkySub'] = lc['apFlux'] - lc['sky'] * aperture_area
    lc['apFluxSkySubErr'] = np.sqrt(lc['apFluxErr'] ** 2 + (lc['skyErr'] * aperture_area) ** 2)

    # Convert from Janskys to erg/sec/cm^2/A
    wavelength = np.vectorize(filters.get)(lc['filterID'])
    lc['apFlux_cgs'] = lc['apFlux'] * 1e-23 * 3e8 / (wavelength**2 * 1e-10)
    lc['apFluxErr_cgs'] = lc['apFluxErr'] * 1e-23 * 3e8 / (wavelength**2 * 1e-10)
    lc['apFluxSkySub_cgs'] = lc['apFluxSkySub'] * 1e-23 * 3e8 / (wavelength**2 * 1e-10)
    lc['apFluxSkySubErr_cgs'] = lc['apFluxSkySubErr'] * 1e-23 * 3e8 / (wavelength**2 * 1e-10)

    return lc


def import_swift_lc(sn, sn_info):
    """
    Imports light curve file from an external source (e.g. Swift)
    """

    # Read CSV
    lc = pd.read_csv(EXTERNAL_LC_DIR / Path('%s_uvotB15.1.dat' % sn), sep='\s+',
            names=['Filter', 'MJD', 'Mag', 'MagErr', '3SigMagLim', '0.98SatLim', 
            'Rate', 'RateErr', 'Ap', 'Frametime', 'Exp', 'Telapse'], comment='#')
    # Remove limits
    lc = lc[pd.notna(lc['Mag'])]
    # lc = lc[(lc['telescope'] == 'Swift') & (lc['instrument'] == 'UVOT') & (lc['upperlimit'] == 'F')]

    # Add days relative to discovery date
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
    lc['t_delta'] = lc['MJD'] - disc_date.mjd

    # Correct epoch for stretch factor
    lc['t_delta_rest'] = 1 / (1 + sn_info.loc[sn, 'z']) * lc['t_delta']

    # Convert CPS to flux
    lc['flux'], lc['flux_err'] = swift_cps2flux(lc['Rate'], lc['RateErr'], lc['Filter'])

    # AB apparent and absolute magnitudes
    # dist = sn_info.loc[sn, 'pref_dist']
    # dist_err = sn_info.loc[sn, 'pref_dist_err']
    # z = sn_info.loc[sn, 'z']
    # lc['ab_mag'], lc['ab_mag_err'] = swift_vega2ab(
    #         lc['magnitude'], lc['e_magnitude'], lc['band'])
    # lc['absolute_mag'], lc['absolute_mag_err'] = absolute_mag_err(
    #         lc['ab_mag'], lc['ab_mag_err'], dist, dist_err)

    # Convert to CPS, flux, and luminosity
    # lc['cps'], lc['cps_err'] = swift_mag2cps(lc['ab_mag'], lc['ab_mag_err'], lc['band'])
    # lc['flux'], lc['flux_err'] = swift_cps2flux(lc['cps'], lc['cps_err'], lc['band'])
    # lc['luminosity'], lc['luminosity_err'] = absolute_luminosity_err(
            # lc['flux'], lc['flux_err'], dist, dist_err, z)

    return lc


################################################################################
## FITS data
################################################################################

def get_fits_files(fits_dir, osc=[]):
    """
    Returns list of FITS files in given data directory; limits to SNe listed in
    OSC reference table, if given
    Inputs:
        fits_dir (Path or str): parent directory for FITS files
        osc (DataFrame): Open Supernova Catalog reference info
    Outputs:
        fits_list (list): list of full FITS file paths
    """
    if type(fits_dir) == str:
        fits_dir = Path(fits_dir) 
    fits_list = [f for f in fits_dir.glob('**/*.fits.gz')]
    if len(osc) > 0:
        fits_list = [f for f in fits_list if fits2sn(f, osc) in osc.index]
    return fits_list


# Convert FITS file name to SN name, as listed in OSC sheet
# Required because Windows doesn't like ':' in file names
def fits2sn(fits_file, osc):
    # Pull SN name from fits file name
    sn_name = '-'.join(fits_file.name.split('-')[:-1])
    # '_' may represent either ':' or ' ' (thanks Windows)
    sn_name = sn_name.replace('_', ' ')
    try:
        osc.loc[sn_name]
    except KeyError as e:
        sn_name = sn_name.replace(' ', ':')
    return sn_name


# Convert SN name to FITS file name
def sn2fits(sn, band=None):
    fits_name = sn.replace(' ','_')
    if (platform.system() == 'Windows') or ('Microsoft' in platform.release()):
        fits_name = fits_name.replace(':','_')
    if band:
        return fits_name + '-' + band + '.fits.gz'
    else:
        return fits_name + '-FUV.fits.gz', fits_name + '-NUV.fits.gz'


class SN:
    def __init__(self, name, osc):
        self.name = name
        disc_date = osc.loc[name, 'Disc. Date']
        self.disc_date = Time(str(disc_date), format='iso', out_subfmt='date')
        max_date = osc.loc[name, 'Max Date']
        if pd.notna(max_date):
            self.max_date = Time(str(max_date), format='iso', out_subfmt='date')
        # self.mmax = osc.loc[name, 'mmax']
        self.host = osc.loc[name, 'Host Name']
        self.ra = Angle(osc.loc[name, 'R.A.'] + ' hours')
        self.dec = Angle(osc.loc[name, 'Dec.'] + ' deg')
        self.z = osc.loc[name, 'z']
        self.type = osc.loc[name, 'Type']
        self.refs = osc.loc[name, 'References'].split(',')


class Fits:
    def __init__(self, fits_file):
        with fits.open(fits_file) as hdu:
            self.header = hdu[0].header
            self.data = hdu[0].data
        self.band = fits_file.name.split('-')[-1].split('.')[0]
        self.path = fits_file
        self.filename = fits_file.name
        # exposure times (array for single image is 2D)
        if self.header['NAXIS'] == 2:
            self.epochs = 1
            expts = [self.header['EXPTIME']]
            tmeans = np.array([(self.header['EXPEND'] + self.header['EXPSTART']) / 2])
        else:
            self.epochs = self.header['NAXIS3']
            expts = [self.header['EXPT'+str(i)] for i in range(self.epochs)]
            tmeans = np.array([self.header['TMEAN'+str(i)] for i in range(self.epochs)])
        self.expts = np.array(expts)
        self.tmeans = Time(np.array(tmeans), format='gps')
        self.wcs = WCS(self.header)
        # RA and Dec are given in degrees
        self.ra = Angle(str(self.header['CRVAL1'])+'d')
        self.dec = Angle(str(self.header['CRVAL2'])+'d')

