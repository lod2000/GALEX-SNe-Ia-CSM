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

import multiprocessing as mp
from itertools import repeat
from functools import partial

SIGMA = 3
OVERWRITE = True

bands = ['FUV', 'NUV']
colors = {'FUV': 'm', 'NUV': 'b'}
styles = {'FUV': 'D', 'NUV': 'o'}


def main():

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
    if BG_FILE.is_file():
        BG_FILE.unlink()

    # Initialize output DataFrames
    # flagged_points = []    
    # total_points = 0
    detected_sne = []
    detections = []
    nondetections = []
    # bg_list = []
    # bg_loc = []
    # exptimes = np.array([])

    # Data flags are in binary
    # flags = [int(2 ** n) for n in range(0,10)]

    print('Searching for detections...')

    with mp.Pool() as pool:
        detected_sne = list(tqdm(
            pool.imap(partial(detect_sn, sn_info=sn_info), sn_info.index, chunksize=10), 
            total=len(sn_info.index)
        ))

    # for sn in detected_sn:
    #     if len(sn) > 0:
    #         detected_sne += sn
    detected_sne = [band for sn in detected_sne for band in sn if len(sn) > 0]
    # print('Plotting light curves...')
    # for sn in tqdm(sn_info.index):
    #     detected_sne += detect_sn(sn, sn_info)

    # List of SNe with detections
    detected_sne = pd.DataFrame(detected_sne, columns=['Name', 'Band', 'Max Sigma'])
    utils.output_csv(detected_sne, 'out/detections.csv', index=False)

    # Limit plot
    # fig = plt.figure(figsize=(11, 7))
    # ax = fig.add_axes([0.1, 0.1, 0.6, 0.8])

    # print('Plotting nondetection limits...')
    # bins = np.arange(-2735, 3105, 365)
    # for lc in tqdm(nondetections):
    #     # lc['group'] = ((lc['t_delta'] + 3100) / 365).astype(int)
    #     group_max = [lc[lc['group'] == g]['luminosity_hostsub_err'].max() for g in np.arange(17)]
    #     ax.scatter(lc['t_delta'], lc['luminosity_hostsub_err'] * SIGMA, 
    #             marker=11, color=colors[band], s=20, alpha=0.1)

    # print('Plotting detections...')
    # for lc in tqdm(detections):
    #     sn = lc['name'].iloc[0]
    #     band = lc['band'].iloc[0]
    #     ax.errorbar(lc['t_delta'], lc['luminosity_hostsub'], linestyle='none', 
    #             yerr=lc['luminosity_hostsub_err'], marker=styles[band], ms=4, 
    #             elinewidth=1, label='%s (%s)' % (sn, band))

    # ax.set_xlim((DT_MIN, None))
    # ax.set_xlabel('Time since discovery [days]')
    # ax.set_ylabel('Luminosity [erg s$^{-1}$ Å$^{-1}$]')
    # ax.set_yscale('log')
    # # ax.set_ylim((1e20, 1e30))
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    # plt.savefig('out/limits.png', dpi=500, bbox_inches='tight')
    # plt.show()


def detect_sn(sn, sn_info):

    detected_sne = []
    
    short_name = Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_short.png')
    full_name = Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_full.png')
    make_plot = (not full_name.is_file()) or OVERWRITE

    if make_plot:
        # Initialize plot
        fig, ax = plt.subplots()
        xmin = xmax = 0

    for band in bands:
        # Import light curve
        try:
            lc = full_import(sn, band, sn_info)
        except FileNotFoundError:
            continue

        # Detect if 3 points above 3 sigma, or 1 point above 5 sigma
        threshold = 5
        lc['sigma_above'] = lc['flux_hostsub'] / lc['flux_hostsub_err']
        lc.insert(0, 'name', np.array([sn] * len(lc.index)))
        lc.insert(1, 'band', np.array([band] * len(lc.index)))
        if len(lc[lc['sigma_above'] >= 3].index) >= 3:
            detected_sne.append([sn, band, np.max(lc['sigma_above'])])
            threshold = 3
        elif len(lc[lc['sigma_above'] >= 5].index) >= 1:
            detected_sne.append([sn, band, np.max(lc['sigma_above'])])
        detections = lc[lc['sigma_above'] >= threshold].index
            # detections.append(lc[lc['sigma_above'] > 5])
            # nondetections.append(lc[lc['sigma_above'] < 5])
        # else:
            # nondetections.append(lc)
        # nondetections = lc[lc['sigma_above'] < threshold]

        if make_plot:
            # Host background & systematic error
            bg, bg_err, sys_err = get_background(lc, band, 'flux_bgsub')
            # 
            fig, ax = plot_band(fig, ax, lc, band, bg, bg_err, 
                    color=colors[band], marker=styles[band], detections=detections)
            xmin = min((xmin, np.min(lc['t_delta'])))
            xmax = max((xmax, np.max(lc['t_delta'])))

    # Configure plot
    if make_plot:
        if xmax > 0 and len(lc.index) > 0:
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

    return detected_sne


def plot_band(fig, ax, lc, band, bg, bg_err, marker='', color='', detections=[]):

    # Plot background average of epochs before discovery
    bg_alpha = 0.2
    plt.axhline(bg, 0, 1, color=color, alpha=0.5, linestyle='--', 
            linewidth=1)
    plt.axhspan(ymin=(bg - SIGMA * bg_err), ymax=(bg + SIGMA * bg_err), 
            color=color, alpha=bg_alpha)

    # Plot detections
    lc_det = lc.loc[detections]
    ax.errorbar(lc_det['t_delta'], lc_det['flux_bgsub'], 
            yerr=lc_det['flux_bgsub_err_total'], linestyle='none', 
            marker='o', ms=5, fillstyle='full',
            elinewidth=1, c=color, label='%s detections' % band
    )


    # Plot nondetections
    lc_non = lc[~lc.index.isin(detections)]
    ax.errorbar(lc_non['t_delta'], lc_non['flux_bgsub'], 
            yerr=lc_non['flux_bgsub_err_total'], linestyle='none', 
            marker='o', ms=5, fillstyle='none',
            elinewidth=1, c=color, label='%s nondetections' % band
    )

    return fig, ax


if __name__ == '__main__':
    main()