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

LC_DIR = Path('/mnt/d/GALEXdata_v6/LCs/')
FITS_DIR = Path('/mnt/d/GALEXdata_v6/fits/')
DETRAD_CUT = 0.55 # deg


def main():

    fits_info = pd.read_csv('out/fitsinfo.csv', index_col='Name')
    sn_info = pd.read_csv('out/sninfo.csv', index_col='Name')
    osc = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')
    ned = pd.read_csv('out/scraped_table.csv', index_col='name')

    flagged_points = []    
    total_points = 0

    bg_list = []
    bg_loc = []

    for sn in tqdm(sn_info.index[0:1]):
        sn = 'SN2007on'

        # Initialize plot
        fig, ax = plt.subplots()
        xlim = (-50, 1000)
        disc_date = Time(sn_info.loc[sn, 'Disc. Date'], format='iso')

        bands = ['FUV', 'NUV']
        colors = ['purple', 'blue']
        marker_styles = ['D', 'o']

        # Show FUV and NUV data on same plot
        for band, color, marker in zip(bands, colors, marker_styles):
            # Import light curve file, if it exists
            try:
                lc = get_lc_data(sn, band, sn_info, ned)
            except FileNotFoundError:
                continue

            # Get host background levels & errors
            bg, bg_err, sys_err = get_background(lc)
            bg_list.append([bg, bg_err, sys_err])
            bg_loc.append((sn, band))

            # Add systematic error in quadrature with statistical
            lc['luminosity_err'] = np.sqrt(lc['luminosity_err'] ** 2 + sys_err ** 2)

            # Background average of epochs before before
            ax.axhline(y=bg_err, alpha=0.8, color=color, label=band+' host background')

            # Convert negative luminosities into upper limits
            lc_lims = lc[lc['luminosity'] < 0]
            ax.scatter(lc_lims['t_delta'], 3 * lc_lims['luminosity_err'] - bg, 
                    marker='v', color=color, label=band+' 3σ limit')

            # Plot luminosities
            lc_data = lc[lc['luminosity'] > 0]
            markers, caps, bars = ax.errorbar(lc_data['t_delta'], lc_data['luminosity'] - bg, 
                    yerr=lc_data['luminosity_err'], linestyle='none', marker=marker, ms=4,
                    elinewidth=1, c=color, label=band
            )
            [bar.set_alpha(0.8) for bar in bars]

        # Configure plot
        ax.set_xlabel('Time since discovery [days]')
        ax.set_xlim(xlim)
        ax.set_ylabel('L_SN - L_host [erg s^-1 Å^-1]')
        ax.set_ylim(0, None)
        plt.legend()
        fig.suptitle(sn)
        plt.savefig(Path('lc_plots/' + sn.replace(':','_') + '.png'))
        plt.show()

    # Output DataFrame of background, bg error, and sys error
    bg_midx = pd.MultiIndex.from_tuples(bg_loc, names=['Name', 'Band'])
    bg_df = pd.DataFrame(bg_list, columns=['Background', 'Background Error', 
            'Systematic Error'], index=bg_midx)
    utils.output_csv(bg_df, 'out/lc_backgrounds.csv')

    #     if len(nuv_lc) > 0:
    #         nuv_bg, nuv_bg_err, nuv_sys_err = get_background(nuv_lc, disc_date)
    #         sys_err.loc[sn, 'NUV Background'] = nuv_bg
    #         sys_err.loc[sn, 'NUV Systematic Error'] = nuv_sys_err
    #         nuv_lc['luminosity_err'] = np.sqrt(nuv_lc['luminosity_err'] ** 2 + nuv_sys_err ** 2)
    #         flags.append(nuv_lc[nuv_lc['luminosity'] > nuv_bg + nuv_bg_err])
    #     if len(fuv_lc) > 0:
    #         fuv_bg, fuv_bg_err, fuv_sys_err = get_background(fuv_lc, disc_date)
    #         sys_err.loc[sn, 'FUV Background'] = fuv_bg
    #         sys_err.loc[sn, 'FUV Systematic Error'] = fuv_sys_err
    #         fuv_lc['luminosity_err'] = np.sqrt(fuv_lc['luminosity_err'] ** 2 + fuv_sys_err ** 2)
    #         flags.append(fuv_lc[fuv_lc['luminosity'] > fuv_bg + fuv_bg_err])

    # flags = pd.concat(flags)
    # flags.to_csv('out/notable.csv')
    # sys_err.to_csv('out/sys_err.csv')


def count_flags(lc):
    """
    Counts the number of points labeled with each flag in a particular light curve
    Also, fix lc files with duplicated information
    Input:
        lc (DataFrame): light curve table
    Output:
        flag_counts (Series): number of data points by flag label
    """

    flags = [int(2 ** n) for n in range(0,10)]
    lc_files = str(LC_DIR) + '/' + fits_info['File'].str.split('.', expand=True)[0] + '.csv'
    lc_files = lc_files.apply(Path)
    low_exptime = []

    print('Reading light curve files...')
    for lc_file in tqdm(lc_files):

        # Count points with only flag 4
        low_exptime += lc[lc['flags'] == 4]['exptime'].tolist()

        # Count total data points
        total_points += len(lc)
        # Count number of data points with each flag
        flag_count = [len(lc[lc['flags'] & f > 0]) for f in flags]
        flagged_points.append(flag_count)

    flagged_points = pd.DataFrame(flagged_points, index=lc_files, columns=flags, dtype=int)
    utils.output_csv(flagged_points, 'out/flagged_points.csv')
    print('Flag counts:')
    print(flagged_points.sum(0))
    print('Flag count fraction:')
    print(flagged_points.sum(0) / total_points)
    print('Total data points: %s' % total_points)
    print('Points with low exptime: mean %s, std %s, count %s' % (np.mean(low_exptime), np.std(low_exptime), len(low_exptime)))

    return flagged_points.sum(0)


def get_background(lc):

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


def get_lc_data(sn, band, sn_info, ned):
    """
    Imports light curve file for specified SN and band. Cuts points with bad
    flags or sources outside detector radius, and also fixes duplicated headers.
    Inputs:
        sn (str): SN name
        band (str): 'FUV' or 'NUV'
        sn_info (DataFrame)
        ned (DataFrame)
    Output:
        lc (DataFrame): light curve table
    """

    # Get name of light curve file
    fits_name = utils.sn2fits(sn, band)
    lc_file = LC_DIR / Path(fits_name.split('.')[0] + '.csv')
    # Discovery date
    disc_date = Time(sn_info.loc[sn, 'Disc. Date'], format='iso')

    # Read light curve data
    lc = pd.read_csv(lc_file)

    # Find duplicated headers, if any, and remove all duplicated material
    # then fix original file
    dup_header = lc[lc['t0'] == 't0']
    if len(dup_header) > 0:
        lc = lc.iloc[0:dup_header.index[0]]
        lc.to_csv(lc_file, index=False)
    lc = lc.astype(float)
    lc['flags'] = lc['flags'].astype(int)

    # Weed out bad flags
    bad_flags = (1 | 2 | 16 | 64 | 128 | 512)
    lc = lc[lc['flags'] & bad_flags == 0]

    # Cut sources outside detector radius
    plate_scale = 6 # as/pixel
    detrad_cut_px = DETRAD_CUT * 3600 / plate_scale
    lc = lc[lc['detrad'] < detrad_cut_px]

    # Convert dates to MJD
    lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd
    lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd
    # Convert measured fluxes to absolute luminosities
    lc['luminosity'] = absolute_luminosity(sn, lc['flux_bgsub'], ned)
    lc['luminosity_err'] = absolute_luminosity(sn, lc['flux_bgsub_err'], ned)

    return lc


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