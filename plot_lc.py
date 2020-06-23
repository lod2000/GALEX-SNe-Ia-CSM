import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time

fits_dir = Path('/mnt/d/GALEXdata_v5/fits/')
fits_file = fits_dir / Path('SN2007on-NUV.fits.gz')
lc_dir = Path('/mnt/d/GALEXdata_v5/LCs/')
lc_file = lc_dir / Path('SN2007on-NUV.csv')
lc = pd.read_csv(lc_file)

fits_info = pd.read_csv('out/fitsinfo.csv', index_col='File')
fits = fits_info.loc[fits_file.name]

tshift = 0
lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd
lc['mcat'] = pd.notna(lc['mag_mcatbgsub'])

t = lc.loc[~lc['mcat'], 't_mean_mjd'] - tshift
mag = lc.loc[~lc['mcat'], 'mag_bgsub']
mag_err_lower = lc.loc[~lc['mcat'], 'mag_bgsub_err_2'].dropna()
mag_err_lower_missing = lc.loc[(pd.isna(lc['mag_bgsub_err_2'])) & (~lc['mcat']), 'mag_bgsub_err_1']
mag_err_lower = pd.concat([mag_err_lower, mag_err_lower_missing]).sort_index()
mag_err_upper = lc.loc[~lc['mcat'], 'mag_bgsub_err_1']
mag_marker = 'o'
plt.errorbar(t, mag, yerr=[mag_err_lower, mag_err_upper], marker=mag_marker, linestyle='none', c='r')

t = lc.loc[lc['mcat'], 't_mean_mjd'] - tshift
mcat = lc.loc[lc['mcat'], 'mag_mcatbgsub']
mcat_err_lower = lc.loc[lc['mcat'], 'mag_mcatbgsub_err_2']
mcat_err_upper = lc.loc[lc['mcat'], 'mag_mcatbgsub_err_1']
mcat_marker = 'D'
plt.errorbar(t, mcat, yerr=[mcat_err_lower, mcat_err_upper], marker=mcat_marker, linestyle='none', c='purple')

plt.gca().invert_yaxis()
plt.show()