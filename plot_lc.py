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

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
    osc = pd.read_csv('ref/osc.csv', index_col='Name')

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
        dt_min = -30
        dt_max = 1000
        xmax = 0
        xmin = 0

        bands = ['FUV', 'NUV']
        colors = ['m', 'b']
        marker_styles = ['D', 'o']

        # Show FUV and NUV data on same plot
        for band, color, marker in zip(bands, colors, marker_styles):
            # Import light curve file, if it exists
            try:
                lc, flag_count = get_lc_data(sn, band, sn_info)
            except FileNotFoundError:
                continue

            # Skip if no useful data points found
            if len(lc.index) == 0:
                continue

            # Get host background levels & errors
            bg, bg_err, sys_err = get_background(lc)
            bg_lum, bg_err_lum, sys_err_lum = absolute_luminosity(sn, np.array([bg, bg_err, sys_err]), sn_info)
            bg_list.append([bg, bg_err, sys_err])
            bg_loc.append((sn, band))

            lc['luminosity_err'] = np.sqrt(lc['luminosity_err'] ** 2 + sys_err_lum ** 2)
            lc['flux_bgsub_err'] = np.sqrt(lc['flux_bgsub_err'] ** 2 + sys_err ** 2)
            lc['luminosity_hostsub'] = lc['luminosity'] - bg_lum
            lc['flux_hostsub'] = lc['flux_bgsub'] - bg

            # Detect points above background error
            lc_det = lc[(lc['flux_bgsub'] - lc['flux_bgsub_err'] > bg + 2 * bg_err) 
                    & (lc['t_delta'] > dt_min) & (lc['t_delta'] < dt_max)]
            n_det = len(lc_det.index)
            lc_det.insert(0, 'name', np.array([sn] * n_det))
            lc_det.insert(1, 'band', np.array([band] * n_det))
            detections.append(lc_det)

            # Count number of data points with each flag
            flagged_points.append(flag_count)
            # Count total data points
            total_points += len(lc)

            xmax = max((xmax, np.max(lc['t_delta'])))
            xmin = min((xmin, np.min(lc['t_delta'])))

            # Plot background average of epochs before discovery
            # ax.axhline(y=bg_err_lum, alpha=0.8, color=color, label=band+' host 2σ background')
            ax.axhline(bg, 0, 1, color=color, alpha=0.5, linestyle='--', 
                linewidth=1, label=band+' host background')
            ax.fill_between(x=[-4000, 4000], y1=bg - 2 * bg_err, y2=bg + 2 * bg_err, 
                    color=color, alpha=0.2, label=band+' host 2σ')

            # Convert negative luminosities into upper limits and plot
            # lc_lims = lc[lc['luminosity'] < 0]
            # sigma = 3
            # ax.scatter(lc_lims['t_delta'], sigma * lc_lims['luminosity_err'] - bg_lum, 
            #         marker='v', color=color, label='%s %sσ limit' % (band, sigma))

            # Plot fluxes
            # lc_data = lc[lc['luminosity'] > 0]
            markers, caps, bars = ax.errorbar(lc['t_delta'], lc['flux_bgsub'], 
                    yerr=lc['flux_bgsub_err'], linestyle='none', marker=marker, ms=4,
                    elinewidth=1, c=color, label=band+' flux'
            )
            [bar.set_alpha(0.8) for bar in bars]

        # Configure plot
        if len(lc.index) > 0:
            ax.set_xlabel('Time since discovery [days]')
            ax.set_xlim((xmin - 50, xmax + 50))
            ax.set_ylabel('Flux [erg s^-1 Å^-1 cm^-2]')
            # ax.set_ylim(0, None)
            plt.legend()
            fig.suptitle(sn)
            plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_full.png'))
            in_range = lc[(lc['t_delta'] > dt_min) & (lc['t_delta'] < dt_max)]
            if len(in_range.index) > 0:
                xlim = (in_range['t_delta'].iloc[0]-20, in_range['t_delta'].iloc[-1]+20)
                ax.set_xlim(xlim)
                plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_short.png'))
            # plt.show()
        plt.close()

    # Output DataFrame of background, bg error, and sys error (flux)
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

    before = lc[lc['t_delta'] < -30]
    data = np.array(before['flux_bgsub'])
    err = np.array(before['flux_bgsub_err'])
    # Need >1 point before discovery to add
    if len(before.index) > 1:
        # Initialize reduced chi-square, sys error values
        rcs = 2
        sys_err = 0
        sys_err_step = np.nanmean(err) * 0.1
        # Reduce RCS to 1 by adding systematic error in quadrature
        while rcs > 1:
            # Combine statistical and systematic error
            new_err = np.sqrt(err ** 2 + sys_err ** 2)
            # Determine background from weighted average of data before discovery
            weighted_stats = DescrStatsW(data, weights=1/new_err**2, ddof=0)
            bg = weighted_stats.mean
            bg_err = weighted_stats.std
            # Reduced chi squared test of data vs background
            rcs = utils.redchisquare(data, np.full(data.size, bg), new_err, n=0)
            # Increase systematic error for next iteration, if necessary
            sys_err += sys_err_step
    else:
        # TODO improve sys error estimate (from gPhoton)
        bg = lc.reset_index(drop=True).loc[0,'flux_bgsub']
        bg_err = lc.reset_index(drop=True).loc[0,'flux_bgsub_err']
        sys_err = 0.07 * bg

    return bg, bg_err, sys_err


def get_lc_data(sn, band, sn_info):
    """
    Imports light curve file for specified SN and band. Cuts points with bad
    flags or sources outside detector radius, and also fixes duplicated headers.
    Inputs:
        sn (str): SN name
        band (str): 'FUV' or 'NUV'
        sn_info (DataFrame)
    Output:
        lc (DataFrame): light curve table
    """

    # Get name of light curve file
    fits_name = utils.sn2fits(sn, band)
    lc_file = LC_DIR / Path(fits_name.split('.')[0] + '.csv')
    # Discovery date
    disc_date = Time(sn_info.loc[sn, 'disc_date'], format='iso')

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
    flags = [int(2 ** n) for n in range(0,10)]
    flag_count = [len(lc[lc['flags'] & f > 0]) for f in flags]
    fatal_flags = (1 | 2 | 4 | 16 | 64 | 128 | 512)
    lc = lc[lc['flags'] & fatal_flags == 0]

    # Cut sources outside detector radius
    plate_scale = 6 # as/pixel
    detrad_cut_px = DETRAD_CUT * 3600 / plate_scale
    lc = lc[lc['detrad'] < detrad_cut_px]

    # Cut ridiculous flux values
    lc = lc[np.abs(lc['flux_bgsub']) < 1]

    # Convert dates to MJD
    lc['t_mean_mjd'] = Time(lc['t_mean'], format='gps').mjd
    lc['t_delta'] = lc['t_mean_mjd'] - disc_date.mjd
    # Convert measured fluxes to absolute luminosities
    lc['luminosity'] = absolute_luminosity(sn, lc['flux_bgsub'], sn_info)
    lc['luminosity_err'] = absolute_luminosity(sn, lc['flux_bgsub_err'], sn_info)

    return lc, flag_count


def absolute_mag(sn, mags, sn_info):
    """
    Converts apparent magnitudes to absolute magnitudes based on NED results
    Inputs:
        sn (str): SN name
        mags (Array-like): apparent magnitudes
        sn_info (DataFrame): includes NED scrape results
    Outputs:
        absolute magnitudes (Array); full of nan if no h_dist is found
    """

    h_dist = sn_info.loc[sn, 'h_dist'] # Mpc
    if pd.notna(h_dist):
        mod = 5 * np.log10(h_dist * 1e6) - 5 # distance modulus
        return mags - mod
    else:
        return np.full(list(mags), np.nan)


def absolute_luminosity(sn, fluxes, sn_info):
    """
    Converts measured fluxes to absolute luminosities based on NED results
    Inputs:
        sn (str): SN name
        mags (Array-like): measured fluxes
        sn_info (DataFrame): includes NED scrape results
    Outputs:
        absolute luminosities (Array)
    """

    h_dist = sn_info.loc[sn, 'h_dist'] # Mpc
    h_dist_cm = h_dist * 3.08568e24 # cm
    luminosities = 4 * np.pi * h_dist_cm**2 * fluxes
    return luminosities


if __name__ == '__main__':
    main()