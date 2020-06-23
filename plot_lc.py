import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time

fits_name = 'SN2007on-NUV.fits.gz'

fits_dir = Path('/mnt/d/GALEXdata_v5/fits/')
fits_file = fits_dir / Path(fits_name)
lc_dir = Path('/mnt/d/GALEXdata_v5/LCs/')
lc_file = lc_dir / Path(fits_name.split('.')[0] + '.csv')
lc = pd.read_csv(lc_file)

fits_info = pd.read_csv('out/fitsinfo.csv', index_col='File')
fits = fits_info.loc[fits_file.name]

tshift = 0
lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd
lc['mcat'] = pd.notna(lc['mag_mcatbgsub'])

fig, ax = plt.subplots()

# Magnitudes without MCAT errors
t = lc.loc[~lc['mcat'], 't_mean_mjd'] - tshift
mag = lc.loc[~lc['mcat'], 'mag_bgsub']
mag_err_lower = lc.loc[~lc['mcat'], 'mag_bgsub_err_2'].dropna()
mag_err_lower_missing = lc.loc[(pd.isna(lc['mag_bgsub_err_2'])) & (~lc['mcat']), 'mag_bgsub_err_1']
mag_err_lower = pd.concat([mag_err_lower, mag_err_lower_missing]).sort_index()
mag_err_upper = lc.loc[~lc['mcat'], 'mag_bgsub_err_1']
mag_marker = 'o'
markers, caps, bars = ax.errorbar(t, mag, yerr=[mag_err_lower, mag_err_upper], 
        marker=mag_marker, linestyle='none', ms=4, elinewidth=1)
[bar.set_alpha(0.8) for bar in bars]

# Magnitudes with MCAT errors
t = lc.loc[lc['mcat'], 't_mean_mjd'] - tshift
mcat = lc.loc[lc['mcat'], 'mag_mcatbgsub']
mcat_err_lower = lc.loc[lc['mcat'], 'mag_mcatbgsub_err_2']
mcat_err_upper = lc.loc[lc['mcat'], 'mag_mcatbgsub_err_1']
mcat_marker = 'D'
markers, caps, bars = ax.errorbar(t, mcat, yerr=[mcat_err_lower, mcat_err_upper], 
        marker=mcat_marker, linestyle='none', ms=6, elinewidth=1)
[bar.set_alpha(0.8) for bar in bars]

# Discovery date
disc_date = Time(fits['Disc. Date'], format='iso')
ax.axvline(disc_date.mjd, 0, 1, c='r', alpha=0.8, linestyle='--')

# Median
before = lc[lc['t_mean'] < (disc_date - 50).gps]
after = lc[lc['t_mean'] < (disc_date + 500).gps]
if len(before) > 0:
    med = np.nanmedian(before['mag_bgsub'])
    med_err_lower = np.sqrt(np.nansum(before['mag_bgsub_err_2']**2))
    med_err_upper = np.sqrt(np.nansum(before['mag_bgsub_err_1']**2))
    med_err = np.nanstd(before['mag_bgsub'])
elif len(after) > 0:
    med = np.nanmedian(after['mag_bgsub'])
    med_err_lower = np.sqrt(np.nansum(after['mag_bgsub_err_2']**2))
    med_err_upper = np.sqrt(np.nansum(after['mag_bgsub_err_1']**2))
    med_err = np.nanstd(after['mag_bgsub'])
else:
    med = lc.loc[0,'mag_bgsub']
    med_err_lower = lc.loc[0, 'mag_bgsub_err_2']
    med_err_upper = lc.loc[0, 'mag_bgsub_err_1']
    med_err = np.nanmax([lc.loc[0, 'mag_bgsub_err_2'], lc.loc[0, 'mag_bgsub_err_1']])

ax.hlines(med, lc['t_mean_mjd'][0], lc.iloc[-1]['t_mean_mjd'], colors=['grey'])
ax.fill_between(lc['t_mean_mjd'], med - 2 * med_err, med + 2 * med_err, 
        alpha=0.2, color='grey')

plt.gca().invert_yaxis()
ax.set_xlabel('Time [MJD]')
ax.set_ylabel('AB Apparent Magnitude')
plt.show()