import pandas as pd
import numpy as np
from pathlib import Path
from astropy.time import Time
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.wcs import WCS


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


# Get list of FITS file names from data directory
def get_fits_files(fits_dir, csv=None):
    if csv:
        fits_list = np.loadtxt(csv, delimiter=',', dtype=str)
        return [fits_dir / sn2fits(f[0], f[1]) for f in fits_list]
    else:
        return [f for f in fits_dir.glob('**/*.fits.gz')]


# Import Open Supernova Catalog csv file
def import_osc(osc_csv):
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
def sn2fits(sn, band):
    return sn.replace(':','_').replace(' ','_') + '-' + band + '.fits.gz'


osc = import_osc(Path('ref/OSC-pre2014-v2-clean.csv'))

class SN:
    def __init__(self, fits_file, ref):
        name = fits2sn(fits_file, ref)
        self.name = name
        # Discovery date is sometimes incomplete
        disc_date = ref.loc[name, 'Disc. Date']
        '''
        if len(str(disc_date).split('-')) < 3:
            self.disc_date = np.nan
        else:
        '''
        self.disc_date = Time(str(disc_date), format='iso', out_subfmt='date')
        #self.disc_date = Time(str(ref.loc[name, 'Disc. Date']), format='iso')
        #self.max_date = Time(str(ref.loc[name, 'Max Date']), format='iso')
        self.mmax = ref.loc[name, 'mmax']
        self.host = ref.loc[name, 'Host Name']
        self.ra = Angle(ref.loc[name, 'R.A.'] + ' hours')
        self.dec = Angle(ref.loc[name, 'Dec.'] + ' deg')
        self.z = ref.loc[name, 'z']
        self.type = ref.loc[name, 'Type']
        #self.refs = ref.loc[name, 'References'].split(',')


class Fits:
    def __init__(self, fits_file):
        with fits.open(fits_file) as hdu:
            self.header = hdu[0].header
            self.data = hdu[0].data
        #self.sn = SN(fits_file)
        self.band = fits_file.name.split('-')[-1].split('.')[0]
        self.path = fits_file
        self.filename = fits_file.name
        # exposure times (array for single image is 2D)
        if self.header['NAXIS'] == 2:
            self.epochs = 1
            expts = [self.header['EXPTIME']]
            tmeans = [(self.header['EXPEND'] + self.header['EXPSTART']) / 2]
        else:
            self.epochs = self.header['NAXIS3']
            expts = [self.header['EXPT'+str(i)] for i in range(self.epochs)]
            tmeans = [self.header['TMEAN'+str(i)] for i in range(self.epochs)]
        self.expts = np.array(expts)
        self.tmeans = Time(np.array(tmeans), format='gps')
        self.wcs = WCS(self.header)
        # RA and Dec are given in degrees
        self.ra = Angle(str(self.header['CRVAL1'])+'d')
        self.dec = Angle(str(self.header['CRVAL2'])+'d')

