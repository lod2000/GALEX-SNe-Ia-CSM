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
# Poole et al. 2007
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


def galex_cps2flux(cps, band):
    # Converts GALEX CPS to flux values
    return np.vectorize(GALEX_FLUX_FACTOR.get)(band) * cps


def galex_cps2mag(cps, band):
    # Converts GALEX CPS to AB magnitudes
    return -2.5 * np.log10(cps) + np.vectorize(GALEX_ZERO_POINT.get)(band)


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


################################################################################
## Importing data
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


# Import Open Supernova Catalog csv file
def import_osc(osc_csv=''):
    if osc_csv == '':
        osc_csv = OSC_FILE
    return pd.read_csv(osc_csv, index_col='Name')


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


# Reduced chi squared statistic
def redchisquare(data, model, sd, n=0):
    chisq = np.sum(((data-model)/sd)**2)
    return chisq/(len(data)-1-n)


class SN:
    def __init__(self, name, osc):
        self.name = name
        disc_date = osc.loc[name, 'Disc. Date']
        self.disc_date = Time(str(disc_date), format='iso', out_subfmt='date')
        max_date = osc.loc[name, 'Max Date']
        self.max_date = Time(str(max_date), format='iso', out_subfmt='date')
        self.mmax = osc.loc[name, 'mmax']
        self.host = osc.loc[name, 'Host Name']
        self.ra = Angle(osc.loc[name, 'R.A.'] + ' hours')
        self.dec = Angle(osc.loc[name, 'Dec.'] + ' deg')
        self.z = osc.loc[name, 'z']
        self.type = osc.loc[name, 'Type']
        self.oscs = osc.loc[name, 'References'].split(',')


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

