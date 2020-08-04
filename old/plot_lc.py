import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import utils
from plot_utils import *
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm


def main():

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
    osc = pd.read_csv('ref/osc.csv', index_col='Name')

    # Initialize output DataFrames
    flagged_points = []    
    total_points = 0
    detections = []
    bg_list = []
    bg_loc = []
    exptimes = np.array([])

    # Data flags are in binary
    flags = [int(2 ** n) for n in range(0,10)]

    print('Plotting light curves...')
    for sn in tqdm(sn_info.index):

        # Initialize plot
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        xmax = 0
        xmin = 0
        handles = labels = []

        bands = ['FUV', 'NUV']
        colors = ['m', 'b']
        marker_styles = ['D', 'o']

        # Show FUV and NUV data on same plot
        for band, color, marker in zip(bands, colors, marker_styles):
            # Import light curve file, if it exists
            try:
                lc, flag_count = get_lc_data(sn, band, sn_info, return_flags=True)
            except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                continue

            # Skip if no useful data points found or not enough background info
            if len(lc.index) == 0 or len(lc[lc['t_delta'] < 0].index) == 0:
                continue

            # Histogram of exposure lengths
            exptimes = np.append(exptimes, lc['exptime'])

            # Get host background levels & errors
            bg, bg_err, sys_err = get_background(lc)
            bg_lum, bg_err_lum, sys_err_lum = absolute_luminosity(sn, np.array([bg, bg_err, sys_err]), sn_info)
            bg_list.append([bg, bg_err, sys_err])
            bg_loc.append((sn, band))

            # Add systematics to lc error bars
            lc['luminosity_err'] = np.sqrt(lc['luminosity_err'] ** 2 + sys_err_lum ** 2)
            lc['flux_bgsub_err'] = np.sqrt(lc['flux_bgsub_err'] ** 2 + sys_err ** 2)
            lc['luminosity_hostsub'] = lc['luminosity'] - bg_lum
            lc['flux_hostsub'] = lc['flux_bgsub'] - bg

            # Detect points above background error
            lc_det = lc[(lc['flux_bgsub'] - lc['flux_bgsub_err'] > bg + 2 * bg_err) 
                    & (lc['t_delta'] > DT_MIN)]
            n_det = len(lc_det.index)
            lc_det.insert(0, 'name', np.array([sn] * n_det))
            lc_det.insert(1, 'band', np.array([band] * n_det))
            lc_det['sigma_above'] = (lc_det['flux_bgsub'] - bg) / bg_err
            lc_det['sigma_above_wgt'] = (lc_det['flux_bgsub'] - bg) / (bg_err * lc_det['flux_bgsub_err'] ** 2)
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

            # Plot fluxes
            # lc_data = lc[lc['luminosity'] > 0]
            markers, caps, bars = ax.errorbar(lc['t_delta'], lc['flux_bgsub'], 
                    yerr=lc['flux_bgsub_err'], linestyle='none', marker=marker, ms=4,
                    elinewidth=1, c=color, label=band+' flux'
            )
            [bar.set_alpha(0.8) for bar in bars]

            handles, labels = ax.get_legend_handles_labels()

        # Configure plot
        if len(labels) > 0 and len(lc.index) > 0:
            ax.set_xlabel('Time since discovery [days]')
            ax.set_xlim((xmin - 50, xmax + 50))
            ax.set_ylabel('Flux [erg s^-1 Å^-1 cm^-2]')
            plt.legend(handles, labels)
            fig.suptitle(sn)
            plt.savefig(Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_full.png'))
            short_range = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < 1000)]
            if len(short_range.index) > 0:
                xlim = (short_range['t_delta'].iloc[0]-20, short_range['t_delta'].iloc[-1]+20)
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

    plot_exptimes(exptimes)
    print('\nExposure times:')
    print('\tOver 30 ks: %s' % exptimes[exptimes >= 29000].shape[0])
    print('\tOver 1500 s: %s' % exptimes[exptimes >= 1400].shape[0])
    print('\tOver 100 s: %s' % exptimes[exptimes >= 90].shape[0])


def plot_exptimes(exptimes):
    plt.hist(exptimes, bins=100)
    plt.show()


if __name__ == '__main__':
    main()