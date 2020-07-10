import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import utils
from lc_utils import *
from tqdm import tqdm

SIGMA = 3
OVERWRITE = True


def main():

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')

    # Initialize output DataFrames
    # flagged_points = []    
    # total_points = 0
    detections = []
    bands = ['FUV', 'NUV']
    colors = {'FUV': 'm', 'NUV': 'b'}
    styles = {'FUV': 'D', 'NUV': 'o'}
    # bg_list = []
    # bg_loc = []
    # exptimes = np.array([])

    # Data flags are in binary
    flags = [int(2 ** n) for n in range(0,10)]

    print('Searching for detections...')
    # print('Plotting light curves...')
    for sn in tqdm(sn_info.index):
        # Initialize plot
        fig, ax = plt.subplots()
        xmin = xmax = 0
        short_name = Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_short.png')
        full_name = Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_full.png')
        make_plot = (not short_name.is_file()) or OVERWRITE

        for band in bands:
            # Import light curve
            try:
                lc = full_import(sn, band, sn_info)
            except FileNotFoundError:
                continue

            detections = find_detections(detections, lc, band, sn)

            xmin = min((xmin, np.min(lc['t_delta'])))
            xmax = max((xmax, np.max(lc['t_delta'])))

            if make_plot:
                # Host background & systematic error
                bg, bg_err, sys_err = get_background(lc, band, 'flux_bgsub')
                fig, ax = plot_band(fig, ax, lc, band, bg, bg_err, 
                        color=colors[band], marker=styles[band])

        # Configure plot
        if xmax > 0 and len(lc.index) > 0 and make_plot:
            ax.set_xlabel('Time since discovery [days]')
            ax.set_xlim((xmin - 50, xmax + 50))
            ax.set_ylabel('Flux [erg s^-1 Å^-1 cm^-2]')
            plt.legend()
            fig.suptitle(sn)
            # Save full figure
            plt.savefig(full_name)
            short_range = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < 1000)]
            # Save short figure
            if len(short_range.index) > 0:
                xlim = (short_range['t_delta'].iloc[0]-20, short_range['t_delta'].iloc[-1]+20)
                ax.set_xlim(xlim)
                plt.savefig(short_name)
        plt.close()


    detections = pd.DataFrame(detections, columns=['Name', 'Band', 'Max Sigma'])
    utils.output_csv(detections, 'out/detections.csv', index=False)


def find_detections(detections, lc, band, sn):

    # Calculate # of sigma above
    lc['sigma_above'] = lc['flux_hostsub'] / lc['flux_hostsub_err']
    # lc['bg_sigma_above'] = lc['flux_hostsub'] / bg_err
    # lc['point_sigma_above'] = lc['flux_hostsub'] / lc['flux_bgsub_err_total']

    # Detect if 3 points above 3 sigma, or 1 point above 5 sigma
    if len(lc[lc['sigma_above'] > 3].index) >= 3 or len(lc[lc['sigma_above'] > 5].index) >= 1:
        detections.append([sn, band, np.max(lc['sigma_above'])])

    return detections


def plot_band(fig, ax, lc, band, bg, bg_err, marker='', color=''):

    # Plot background average of epochs before discovery
    bg_alpha = 0.2
    plt.axhline(bg, 0, 1, color=color, alpha=0.5, linestyle='--', 
            linewidth=1)
    plt.axhspan(ymin=(bg - SIGMA * bg_err), ymax=(bg + SIGMA * bg_err), 
            color=color, alpha=bg_alpha)

    # Plot fluxes
    ax.errorbar(lc['t_delta'], lc['flux_bgsub'], 
            yerr=lc['flux_bgsub_err_total'], linestyle='none', 
            marker='o', ms=5,
            elinewidth=1, c=color, label=band
    )

    return fig, ax


if __name__ == '__main__':
    main()