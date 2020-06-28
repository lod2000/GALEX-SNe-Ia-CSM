import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import utils
from scipy import stats
from tqdm import tqdm

LC_DIR = Path('/mnt/d/GALEXdata_v5/LCs/')
FITS_DIR = Path('/mnt/d/GALEXdata_v5/fits/')


def main():

    fits_info = pd.read_csv('out/fitsinfo.csv', index_col='Name')
    sne = fits_info.index.drop_duplicates()
    ref = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')
    ned = pd.read_csv('out/scraped_table.csv', index_col='name')

    flags = []
    sys_err = pd.DataFrame([], columns=['NUV Background', 'NUV Systematic Error', 'FUV Background', 'FUV Systematic Error'], 
            index=pd.Series(sne, name='Name'))

    for sn in tqdm(sne[0:1]):
        sn = 'SN2007on'
        disc_date = Time(ref.loc[sn, 'Disc. Date'], format='iso')

        nuv_lc = get_lc_data(sn, 'NUV', disc_date, ned)
        fuv_lc = get_lc_data(sn, 'FUV', disc_date, ned)

        nuv_bg = nuv_bg_err = fuv_bg = fuv_bg_err = None

        if len(nuv_lc) > 0:
            nuv_bg, nuv_bg_err, nuv_sys_err = get_background(nuv_lc, disc_date)
            sys_err.loc[sn, 'NUV Background'] = nuv_bg
            sys_err.loc[sn, 'NUV Systematic Error'] = nuv_sys_err
            nuv_lc['luminosity_err'] = np.sqrt(nuv_lc['luminosity_err'] ** 2 + nuv_sys_err ** 2)
            flags.append(nuv_lc[nuv_lc['luminosity'] > nuv_bg + nuv_bg_err])
        if len(fuv_lc) > 0:
            fuv_bg, fuv_bg_err, fuv_sys_err = get_background(fuv_lc, disc_date)
            sys_err.loc[sn, 'FUV Background'] = fuv_bg
            sys_err.loc[sn, 'FUV Systematic Error'] = fuv_sys_err
            fuv_lc['luminosity_err'] = np.sqrt(fuv_lc['luminosity_err'] ** 2 + fuv_sys_err ** 2)
            flags.append(fuv_lc[fuv_lc['luminosity'] > fuv_bg + fuv_bg_err])

        plot_lc(nuv_lc, fuv_lc, sn, disc_date, nuv_bg, nuv_bg_err, fuv_bg, fuv_bg_err)

    flags = pd.concat(flags)
    flags.to_csv('out/notable.csv')
    sys_err.to_csv('out/sys_err.csv')


def get_background(lc, disc_date):

    before = lc[lc['t_delta'] < -50]
    data = np.array(before['luminosity'])
    err = np.array(before['luminosity_err'])
    if len(before) > 1:
        # Determine background from weighted average of data before discovery
        bg = np.average(data, weights=err)
        # bg_err = np.std(data)
        bg_err = np.sqrt(np.sum(err ** 2))
        # Reduced chi squared test of data vs background
        rcs = utils.redchisquare(data, np.full(data.size, bg), err, n=0)
        sys_err = err[0] * 0.1
        # Reduce RCS to 1 by adding systematic error in quadrature
        while rcs > 1:
            sys_err += err[0] * 0.1
            new_err = np.sqrt(err ** 2 + sys_err ** 2)
            rcs = utils.redchisquare(data, np.full(data.size, bg), new_err, n=0)
            bg = np.average(data, weights=new_err)
            bg_err = np.sqrt(np.sum(new_err ** 2))
    else:
        bg = lc.loc[0, 'luminosity']
        bg_err = lc.loc[0, 'luminosity_err']
        sys_err = np.nan

    return bg, bg_err, sys_err


def get_lc_data(sn, band, disc_date, ned):

    # Get name of light curve file
    fits_name = utils.sn2fits(sn, band)
    lc_file = LC_DIR / Path(fits_name.split('.')[0] + '.csv')

    # Read light curve data
    try:
        lc = pd.read_csv(lc_file, dtype={'flags':int})

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
    except FileNotFoundError:
        return []


def plot_lc(nuv_lc, fuv_lc, sn, disc_date, nuv_bg=None, nuv_bg_err=None, fuv_bg=None, fuv_bg_err=None):

    fig, ax = plt.subplots()

    xlim = (-50, 1000)

    # Background median of epochs before or long after discovery
    # ax.hlines(bg, xlim[0], xlim[1], colors=['grey'], label='Background median')
    if nuv_bg:
        ax.fill_between(np.arange(xlim[0], xlim[1]), 0, nuv_bg_err, 
                alpha=0.5, color='blue', label='NUV host background')
    if fuv_bg:
        ax.fill_between(np.arange(xlim[0], xlim[1]), 0, fuv_bg_err, 
                alpha=0.5, color='purple', label='FUV host background')
    # ax.fill_between(lc['t_mean_mjd'], med - 2 * med_err, med + 2 * med_err, 
    #         alpha=0.2, color='grey', label='Background 2σ')

    # Convert negative luminosities into upper limits
    if len(nuv_lc) > 0:
        nuv_lims = nuv_lc[nuv_lc['luminosity'] < 0]
        nuv_lc = nuv_lc[nuv_lc['luminosity'] > 0]
        # NUV fluxes
        markers, caps, bars = ax.errorbar(nuv_lc['t_delta'], nuv_lc['luminosity'] - nuv_bg, 
                yerr=nuv_lc['luminosity_err'], linestyle='none', marker='o', ms=4,
                elinewidth=1, c='blue', label='NUV'
        )
        [bar.set_alpha(0.8) for bar in bars]
        # NUV upper limits
        ax.scatter(nuv_lims['t_delta'], 3 * nuv_lims['luminosity_err'] - nuv_bg, 
                marker='v', color='blue', label='NUV 3σ limit')

    # Convert negative luminosities into upper limits
    if len(fuv_lc) > 0:
        fuv_lims = fuv_lc[fuv_lc['luminosity'] < 0]
        fuv_lc = fuv_lc[fuv_lc['luminosity'] > 0]
        # FUV fluxes
        markers, caps, bars = ax.errorbar(fuv_lc['t_delta'], fuv_lc['luminosity'] - fuv_bg, 
                yerr=fuv_lc['luminosity_err'], linestyle='none', marker='D', ms=4,
                elinewidth=1, c='purple', label='FUV'
        )
        [bar.set_alpha(0.8) for bar in bars]
        # FUV upper limits
        ax.scatter(fuv_lims['t_delta'], 3 * fuv_lims['luminosity_err'] - fuv_bg, 
                marker='v', color='purple', label='FUV 3σ limit')

    # Configure plot
    # plt.gca().invert_yaxis()
    ax.set_xlabel('Time since discovery [days]')
    ax.set_xlim(xlim)
    ax.set_ylabel('L_SN - L_host [erg s^-1 Å^-1]')
    plt.legend()
    fig.suptitle(sn)
    # plt.show()
    plt.savefig('lc_plots/' + sn + '.png')
    plt.close()


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