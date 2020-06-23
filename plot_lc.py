import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import utils
from scipy import stats


def main():

    fits_name = 'SN2007on-NUV.fits.gz'

    fits_dir = Path('/mnt/d/GALEXdata_v5/fits/')
    fits_file = fits_dir / Path(fits_name)
    lc_dir = Path('/mnt/d/GALEXdata_v5/LCs/')
    lc_file = lc_dir / Path(fits_name.split('.')[0] + '.csv')
    lc = pd.read_csv(lc_file)
    band = fits_file.name.split('-')[-1].split('.')[0]

    fits_info = pd.read_csv('out/fitsinfo.csv', index_col='File')
    fits = fits_info.loc[fits_file.name]
    disc_date = Time(fits['Disc. Date'], format='iso')

    lc = get_lc_data(lc, band)
    med, med_err = get_median(lc, disc_date)

    plot_lc(lc, disc_date, med, med_err, fits_name)

    floor = med - 2 * med_err
    flag = lc[(lc['mag_mcatbgsub'] < floor) | (lc['mag_bgsub'] < floor)]
    print(flag)


def get_median(lc, disc_date):

    before = lc[lc['t_mean'] < (disc_date - 50).gps]
    after = lc[lc['t_mean'] > (disc_date + 500).gps]
    if len(before) > 0:
        med = np.nanmedian(before['mag_plot'])
        print(med)
        med_err = np.nanstd(before['mag_plot'])
        wgt_mean = np.average(before['mag_plot'], weights=before['mag_plot_err_1'])
        print(wgt_mean)
        chisquare, p = stats.chisquare(before['mag_plot'], med, ddof=len(before)-1)
        print(chisquare / med)
    elif len(after) > 0:
        med = np.nanmedian(after['mag_plot'])
        med_err = np.nanstd(after['mag_plot'])
    else:
        med = lc.loc[0,'mag_plot']
        med_err = np.nanmax([lc.loc[0, 'mag_plot_err_2'], 
                lc.loc[0, 'mag_plot_err_1']])

    return med, med_err


def get_lc_data(lc, band):

    lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd
    # MCAT magnitudes and errors
    lc['mcat'] = pd.notna(lc['mag_mcatbgsub'])
    lc.loc[lc['mcat'], 'mag_plot'] = lc.loc[lc['mcat'], 'mag_mcatbgsub']
    lc.loc[lc['mcat'], 'mag_plot_err_2'] = lc.loc[lc['mcat'], 'mag_mcatbgsub_err_2']
    lc.loc[lc['mcat'], 'mag_plot_err_1'] = lc.loc[lc['mcat'], 'mag_mcatbgsub_err_1']
    lc.loc[lc['mcat'], 'type'] = 'mcat'
    # Non-MCAT magnitudes and errors
    lc.loc[~lc['mcat'], 'mag_plot'] = lc.loc[~lc['mcat'], 'mag_bgsub']
    lc.loc[~lc['mcat'], 'mag_plot_err_2'] = lc.loc[~lc['mcat'], 'mag_bgsub_err_2']
    # fill missing lower error values
    lc.loc[(pd.isna(lc['mag_bgsub_err_2'])) & (~lc['mcat']), 'mag_plot_err_2'] = 0
    lc.loc[~lc['mcat'], 'mag_plot_err_1'] = lc.loc[~lc['mcat'], 'mag_bgsub_err_1']
    lc.loc[(~lc['mcat']) & (pd.notna(lc['mag_bgsub'])), 'type'] = 'normal'
    # Upper bounds for missing mags
    lc.loc[pd.isna(lc['mag_plot']), 'mag_plot'] = utils.galex_ab_mag(
            3*lc.loc[pd.isna(lc['mag_plot']), 'cps_bgsub_err'], band)
    lc.loc[(~lc['mcat']) & (pd.isna(lc['mag_bgsub'])), 'type'] = 'upperlim'

    return lc


def plot_lc(lc, disc_date, med, med_err, fits_name):

    fig, ax = plt.subplots()

    # Background median of epochs before or long after discovery
    ax.hlines(med, lc['t_mean_mjd'][0], lc.iloc[-1]['t_mean_mjd'], colors=['grey'], 
            label='Background median')
    ax.fill_between(lc['t_mean_mjd'], med - med_err, med + med_err, 
            alpha=0.5, color='grey', label='Background 1σ')
    ax.fill_between(lc['t_mean_mjd'], med - 2 * med_err, med + 2 * med_err, 
            alpha=0.2, color='grey', label='Background 2σ')

    # Upper limits
    upperlim = lc[lc['type'] == 'upperlim']
    ax.scatter(upperlim['t_mean_mjd'], upperlim['mag_plot'], 
            marker='v', c='green', label='3σ detection limit')

    # Magnitudes without MCAT errors
    normal = lc[lc['type'] == 'normal']
    markers, caps, bars = ax.errorbar(normal['t_mean_mjd'], normal['mag_plot'], 
            yerr=[normal['mag_plot_err_1'], normal['mag_plot_err_2']], 
            marker='o', linestyle='none', ms=4, elinewidth=1, c='blue',
            label='Background-sub magnitudes'
    )
    [bar.set_alpha(0.8) for bar in bars]

    # Magnitudes with MCAT errors
    mcat = lc[lc['type'] == 'mcat']
    markers, caps, bars = ax.errorbar(mcat['t_mean_mjd'], mcat['mag_plot'], 
            yerr=[mcat['mag_plot_err_1'], mcat['mag_plot_err_2']], 
            marker='D', linestyle='none', ms=6, elinewidth=1, c='orange',
            label='MCAT background-sub magnitude'
    )
    [bar.set_alpha(0.8) for bar in bars]

    # Discovery date
    ax.axvline(disc_date.mjd, 0, 1, c='r', alpha=0.8, linestyle='--', 
            label='Discovery date')

    # Configure plot
    plt.gca().invert_yaxis()
    ax.set_xlabel('Time [MJD]')
    ax.set_xlim((lc['t_mean_mjd'][0] - 50, lc.iloc[-1]['t_mean_mjd'] + 50))
    ax.set_ylabel('AB Apparent Magnitude')
    plt.legend()
    fig.suptitle(fits_name.split('.')[0])
    plt.show()


if __name__ == '__main__':
    main()