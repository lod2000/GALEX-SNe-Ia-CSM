import numpy as np
import pandas as pd
from pathlib import Path
from astropy.time import Time
import utils
from statsmodels.stats.weightstats import DescrStatsW

LC_DIR = Path('/mnt/d/GALEXdata_v10/LCs/')
FITS_DIR = Path('/mnt/d/GALEXdata_v10/fits/')
EXTERNAL_LC_DIR = Path('external/')
BG_FILE = Path('out/high_bg.csv')
EMPTY_LC_FILE = Path('out/empty_lc.csv')
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
        utils.output_csv(empty_lc, EMPTY_LC_FILE, index=False)
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
    lc['luminosity'], lc['luminosity_err'] = absolute_luminosity_err(
            lc['flux_bgsub'], lc['flux_bgsub_err_total'], dist, dist_err)
    lc['luminosity_hostsub'], lc['luminosity_hostsub_err'] = absolute_luminosity_err(
            lc['flux_hostsub'], lc['flux_hostsub_err'], dist, dist_err)

    # Convert apparent to absolute magnitudes
    lc['absolute_mag'], lc['absolute_mag_err_1'] = absolute_mag_err(
            lc['mag_bgsub'], lc['mag_bgsub_err_1'], dist, dist_err)
    lc['absolute_mag_err_2'] = absolute_mag_err(
            lc['mag_bgsub'], lc['mag_bgsub_err_2'], dist, dist_err)[1]

    return lc, bg, bg_err, sys_err


def galex_flux2mag(flux, band):
    """
    Converts fluxes from GALEX to AB magnitudes
    """

    zero_point = {'FUV': 18.82, 'NUV': 20.08}
    factor = {'FUV': 1.4e-15, 'NUV': 2.06e-16}
    return -2.5 * np.log10(flux / np.vectorize(factor.get)(band)) + np.vectorize(zero_point.get)(band)


def galex_mag2cps(mag, mag_err, band):
    """
    Converts AB magnitudes measured by GALEX into flux
    """

    zero_point = {'FUV': 18.82, 'NUV': 20.08}
    cps = 10 ** (2/5 * (np.vectorize(zero_point.get)(band) - mag))
    cps_err = cps * (2/5) * np.log(10) * mag_err
    return cps, cps_err


def galex_cps2flux(cps, band):
    """
    Converts GALEX CPS to flux values
    """

    conversion = {'FUV': 1.4e-15, 'NUV': 2.06e-16}
    return np.vectorize(conversion.get)(band) * cps


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
            rcs = utils.redchisquare(data, np.full(data.size, bg), new_err, n=0)

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
        utils.output_csv(high_bg, BG_FILE, index=True)
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

    # Read CSV and select only Swift UVOT data
    lc = pd.read_csv(EXTERNAL_LC_DIR / Path('%s_phot.csv' % sn))
    lc = lc[(lc['telescope'] == 'Swift') & (lc['instrument'] == 'UVOT') & (lc['upperlimit'] == 'F')]

    # Add days relative to discovery date
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')
    lc['t_delta'] = lc['time'] - disc_date.mjd

    # AB apparent and absolute magnitudes
    dist = sn_info.loc[sn, 'pref_dist']
    dist_err = sn_info.loc[sn, 'pref_dist_err']
    lc['ab_mag'], lc['ab_mag_err'] = swift_vega2ab(
            lc['magnitude'], lc['e_magnitude'], lc['band'])
    lc['absolute_mag'], lc['absolute_mag_err'] = absolute_mag_err(
            lc['ab_mag'], lc['ab_mag_err'], dist, dist_err)

    # Convert to CPS, flux, and luminosity
    lc['cps'], lc['cps_err'] = swift_mag2cps(lc['ab_mag'], lc['ab_mag_err'], lc['band'])
    lc['flux'], lc['flux_err'] = swift_cps2flux(lc['cps'], lc['cps_err'], lc['band'])
    lc['luminosity'], lc['luminosity_err'] = absolute_luminosity_err(
            lc['flux'], lc['flux_err'], dist, dist_err)

    return lc


def swift_vega2ab(vega_mag, vega_mag_err, band):
    """
    Converts Vega magnitudes from Swift to AB magnitudes, based on the band;
    conversion values from Breeveld et al. 2011
    Inputs:
        vega_mag (float or Array): Swift Vega magnitude
        vega_mag_err (float or Array): Swift Vega magnitude error
        band (str or Array): UVW1, UVM2, or UVW2
    """

    conversion = {'V':-0.01, 'B':-0.13, 'U': 1.02, 'UVW1': 1.51, 'UVM2': 1.69, 'UVW2': 1.73}
    conv_error = {'V': 0.01, 'B': 0.02, 'U': 0.02, 'UVW1': 0.03, 'UVM2': 0.03, 'UVW2': 0.03}
    const = np.vectorize(conversion.get)(band)
    ab_mag_err = np.sqrt(vega_mag_err**2 + np.vectorize(conv_error.get)(band)**2)
    return vega_mag + const, ab_mag_err


def swift_mag2cps(mag, mag_err, band):
    # Zero points from Poole et al. 2007
    zero_point = {'V': 17.89, 'B': 19.11, 'U': 18.34, 'UVW1': 17.49, 
            'UVM2': 16.82, 'UVW2': 17.35}
    zpt_err = {'V': 0.013, 'B': 0.016, 'U': 0.020, 'UVW1': 0.03, 'UVM2': 0.03,
            'UVW2': 0.03}
    diff = np.vectorize(zero_point.get)(band) - mag
    diff_err = np.sqrt(mag_err ** 2 + np.vectorize(zpt_err.get)(band) ** 2)
    cps = 10 ** (2/5 * diff)
    cps_err = cps * (2/5) * np.log(10) * diff_err
    return cps, cps_err


def swift_cps2flux(cps, cps_err, band):
    # Conversion values from Poole et al. 2007
    conversion = {'V': 2.614e-16, 'B': 1.472e-16, 'U': 1.63e-16, 
            'UVW1': 4.3e-16, 'UVM2': 7.5e-16, 'UVW2': 6.0e-16}
    conv_error = {'V': 8.7e-19, 'B': 5.7e-19, 'U': 2.5e-18, 'UVW1': 2.1e-17, 
            'UVM2': 1.1e-16, 'UVW2': 6.4e-17}
    c = np.vectorize(conversion.get)(band)
    c_err = np.vectorize(conv_error.get)(band)
    flux = cps * c
    flux_err = flux * np.sqrt((cps_err/cps)**2 + (c_err/c)**2)
    return flux, flux_err