import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import utils
from scipy import stats

LC_DIR = Path('/mnt/d/GALEXdata_v5/LCs/')
FITS_DIR = Path('/mnt/d/GALEXdata_v5/fits/')


def main():

    fits_info = pd.read_csv('out/fitsinfo.csv', index_col='Name')
    ref = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')
    ned = pd.read_csv('out/scraped_table.csv', index_col='name')

    sn = 'SN2007on'

    disc_date = Time(ref.loc[sn, 'Disc. Date'], format='iso')

    lc_nuv = get_lc_data(sn, 'NUV', disc_date, ned)
    lc_fuv = get_lc_data(sn, 'FUV', disc_date, ned)

    bg, bg_err, err_step = get_background(lc_nuv, disc_date)
    print(err_step)
    lc_fuv['luminosity_err'] = np.sqrt(lc_nuv['luminosity_err'] ** 2 + err_step ** 2)

    plot_lc(lc_nuv, lc_fuv, sn, disc_date, bg, bg_err)

    # floor = med - 2 * med_err
    # flag = lc[(lc['mag_mcatbgsub'] < floor) | (lc['mag_bgsub'] < floor)]
    # print(flag)


def get_background(lc, disc_date):

    before = lc[lc['t_delta'] < -50]
    data = np.array(before['luminosity'])
    err = np.array(before['luminosity_err'])
    if len(before) > 1:
        bg = np.average(data, weights=err)
        bg_err = np.sqrt(np.sum(err ** 2))
        rcs = utils.redchisquare(data, np.full(data.size, bg), err, n=0)
        err_step = err[0] * 0.1
        while rcs > 1:
            err_step += err[0] * 0.1
            new_err = np.sqrt(err ** 2 + err_step ** 2)
            rcs = utils.redchisquare(data, np.full(data.size, bg), new_err, n=0)
        bg = np.average(data, weights=new_err)
        bg_err = np.sqrt(np.sum(new_err ** 2))
    else:
        bg = lc.loc[0, 'luminosity']
        bg_err = lc.loc[0, 'luminosity_err']

    return bg, bg_err, err_step


def get_lc_data(sn, band, disc_date, ned):

    # Get name of light curve file
    fits_name = utils.sn2fits(sn, band)
    lc_file = LC_DIR / Path(fits_name.split('.')[0] + '.csv')

    # Read light curve data
    lc = pd.read_csv(lc_file)
    lc['flags'] = lc['flags'].astype(int)

    # Weed out bad flags
    bad_flags = (1 | 2 | 16 | 64 | 128 | 512)
    lc = lc[lc['flags'] & bad_flags == 0]

    # Convert dates to MJD
    lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd
    lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd
    # Convert measured fluxes to absolute luminosities
    lc['luminosity'] = absolute_luminosity(sn, lc['flux_bgsub'], ned)
    lc['luminosity_err'] = absolute_luminosity(sn, lc['flux_bgsub_err'], ned)

    return lc


def plot_lc(lc_nuv, lc_fuv, sn, disc_date, bg, bg_err):

    fig, ax = plt.subplots()

    xlim = (-1000, 1000)

    # Background median of epochs before or long after discovery
    ax.hlines(bg, xlim[0], xlim[1], colors=['grey'], label='Background median')
    ax.fill_between(np.arange(xlim[0], xlim[1]), bg - bg_err, bg + bg_err, 
            alpha=0.5, color='grey', label='Background 1σ')
    # ax.fill_between(lc['t_mean_mjd'], med - 2 * med_err, med + 2 * med_err, 
    #         alpha=0.2, color='grey', label='Background 2σ')

    # Upper limits
    # upperlim = lc[lc['type'] == 'upperlim']
    # ax.scatter(upperlim['t_mean_mjd'], upperlim['mag_plot'], 
    #         marker='v', c='green', label='3σ detection limit')

    # NUV fluxes
    markers, caps, bars = ax.errorbar(lc_nuv['t_delta'], lc_nuv['luminosity'], 
            yerr=lc_nuv['luminosity_err'], linestyle='none', marker='o', ms=4,
            elinewidth=1, c='blue', label='NUV background-corrected flux'
    )
    [bar.set_alpha(0.8) for bar in bars]

    # FUV fluxes
    markers, caps, bars = ax.errorbar(lc_fuv['t_delta'], lc_fuv['luminosity'], 
            yerr=lc_fuv['luminosity_err'], linestyle='none', marker='D', ms=4,
            elinewidth=1, c='purple', label='FUV background-corrected flux'
    )
    [bar.set_alpha(0.8) for bar in bars]

    # Configure plot
    # plt.gca().invert_yaxis()
    ax.set_xlabel('Time since discovery [days]')
    ax.set_xlim(xlim)
    ax.set_ylabel('Luminosity [erg s^-1 Å^-1]')
    plt.legend()
    fig.suptitle(sn)
    plt.show()


def absolute_mag(sn, mags, ned):
    """
    Converts apparent magnitudes to absolute magnitudes based on NED results
    Inputs:
        sn (str): SN name
        mags (Array-like): apparent magnitudes
        ned (DataFrame): NED scrape results
    Outputs:
        absolute magnitudes (Array); full of nan if no h_dist is found
    """

    h_dist = ned.loc[sn, 'h_dist'] # Mpc
    if pd.notna(h_dist):
        mod = 5 * np.log10(h_dist * 1e6) - 5 # distance modulus
        return mags - mod
    else:
        return np.full(list(mags), np.nan)


def absolute_luminosity(sn, fluxes, ned):
    """
    Converts measured fluxes to absolute luminosities based on NED results
    Inputs:
        sn (str): SN name
        mags (Array-like): measured fluxes
        ned (DataFrame): NED scrape results
    Outputs:
        absolute luminosities (Array); full of nan if no h_dist is found
    """

    h_dist = ned.loc[sn, 'h_dist'] # Mpc
    h_dist_cm = h_dist * 3.08568e24 # cm
    luminosities = 4 * np.pi * h_dist_cm**2 * fluxes
    return luminosities


if __name__ == '__main__':
    main()