import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import utils
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm

LC_DIR = Path('/mnt/d/GALEXdata_v6/LCs/')
FITS_DIR = Path('/mnt/d/GALEXdata_v6/fits/')
DETRAD_CUT = 0.55 # deg


def main():

    fits_info = pd.read_csv('out/fitsinfo.csv', index_col='Name')
    sn_info = pd.read_csv('out/sninfo.csv', index_col='Name')
    osc = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')
    ned = pd.read_csv('out/scraped_table.csv', index_col='name')

    # Initialize output DataFrames
    flagged_points = []    
    total_points = 0
    detections = []
    bg_list = []
    bg_loc = []

    # Data flags are in binary
    flags = [int(2 ** n) for n in range(0,10)]

    print('Plotting light curves...')
    for sn in tqdm(sn_info.index):

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

            # Detect points above background error
            lc_det = lc[lc['luminosity'] > bg + bg_err]
            n_det = len(lc_det.index)
            lc_det.insert(0, 'name', np.array([sn] * n_det))
            lc_det.insert(1, 'band', np.array([band] * n_det))
            detections.append(lc_det)

            # Count number of data points with each flag
            flag_count = [len(lc[lc['flags'] & f > 0]) for f in flags]
            flagged_points.append(flag_count)
            # Count total data points
            total_points += len(lc)

            # Plot background average of epochs before before
            ax.axhline(y=bg_err, alpha=0.8, color=color, label=band+' host background')

            # Convert negative luminosities into upper limits and plot
            lc_lims = lc[lc['luminosity'] < 0]
            sigma = 3
            ax.scatter(lc_lims['t_delta'], sigma * lc_lims['luminosity_err'] - bg, 
                    marker='v', color=color, label='%s %sσ limit' % (band, sigma))

            # Plot luminosities
            lc_data = lc[lc['luminosity'] > 0]
            markers, caps, bars = ax.errorbar(lc_data['t_delta'], lc_data['luminosity'] - bg, 
                    yerr=lc_data['luminosity_err'], linestyle='none', marker=marker, ms=4,
                    elinewidth=1, c=color, label=band
            )
            [bar.set_alpha(0.8) for bar in bars]

        # Configure plot
        if len(lc.index) > 0:
            ax.set_xlabel('Time since discovery [days]')
            ax.set_xlim(xlim)
            ax.set_ylabel('L_SN - L_host [erg s^-1 Å^-1]')
            ax.set_ylim(0, None)
            plt.legend()
            fig.suptitle(sn)
            plt.savefig(Path('lc_plots/' + sn.replace(':','_') + '.png'))
            # plt.show()
        plt.close()

    # Output DataFrame of background, bg error, and sys error
    # Right now outputs luminosity; may want to change to flux later
    bg_midx = pd.MultiIndex.from_tuples(bg_loc, names=['Name', 'Band'])
    bg_df = pd.DataFrame(bg_list, columns=['Background', 'Background Error', 
            'Systematic Error'], index=bg_midx)
    utils.output_csv(bg_df, 'out/lc_backgrounds.csv')

    # Output possible detections
    detections = pd.concat(detections)
    detections.reset_index(inplace=True)
    utils.output_csv(detections, 'out/detections.csv', index=False)

    # Output flag info
    flagged_points = pd.DataFrame(flagged_points, index=bg_midx, columns=flags, dtype=int)
    utils.output_csv(flagged_points, 'out/flagged_points.csv')
    print('Flag counts:')
    print(flagged_points.sum(0))
    print('Flag count fraction:')
    print(flagged_points.sum(0) / total_points)
    print('Total data points: %s' % total_points)


def get_background(lc):
    """
    Calculates the host background for a given light curve. Also calculates the
    systematic error needed to make the reduced chi squared value of the total
    error equal to 1.
    Inputs:
        lc (DataFrame): light curve table
    Outputs:
        bg (float): host background luminosity
        bg_err (float): host background luminosity error
        sys_err (float): systematic error based on reduced chi-squared test
    """

    before = lc[lc['t_delta'] < -50]
    data = np.array(before['luminosity'])
    err = np.array(before['luminosity_err'])
    if len(before) > 1:
        # Determine background from weighted average of data before discovery
        weighted_stats = DescrStatsW(data, weights=err, ddof=0)
        bg = weighted_stats.mean
        bg_err = weighted_stats.std_mean
        # bg = np.average(data, weights=err)
        # bg_err = np.sqrt(np.sum(err ** 2))
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
        # TODO improve sys error estimate (from gPhoton)
        bg = lc.iloc[0]['luminosity']
        bg_err = lc.iloc[0]['luminosity_err']
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