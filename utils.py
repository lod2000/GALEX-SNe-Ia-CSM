import pandas as pd
import numpy as np
from pathlib import Path
import platform
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS

OSC_FILE = Path('ref/osc.csv')


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


# https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
def galex_ab_mag(cps, band):
    const = 18.82 if band=='FUV' else 20.08
    return -2.5 * np.log10(cps) + const


def galex_flux(cps, band):
    factor = 1.4e-15 if band=='FUV' else 2.06e-16
    return factor * cps


def galex_delta_mag(cps, band, exp_time):
    factor = 0.05 if band=='FUV' else 0.027
    return -2.5 * (np.log10(cps) - np.log10(cps + np.sqrt(cps * exp_time + \
            (factor * cps * exp_time) ** 2) / exp_time))


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

