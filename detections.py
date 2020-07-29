#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
from utils import *
from tqdm import tqdm
import argparse

import multiprocessing as mp
from itertools import repeat
from functools import partial

bands = ['FUV', 'NUV']
np.seterr(all='warn')


def main():

    parser = argparse.ArgumentParser(description='Detect SNe in GALEX light curves.')
    parser.add_argument('-s', '--sigma', type=float, default=3, 
            help='number of sigma to plot as host background')
    parser.add_argument('-i', '--info', type=Path, default=Path('ref/sn_info.csv'),
            help='path to sn info csv file', metavar='file.csv')
    parser.add_argument('-o', '--overwrite', action='store_true',
            help='re-generate all plots')
    args = parser.parse_args()

    sn_info = pd.read_csv(args.info, index_col='name')
    if BG_FILE.is_file():
        BG_FILE.unlink()
    if EMPTY_LC_FILE.is_file():
        EMPTY_LC_FILE.unlink()

    # Initialize output DataFrames
    detected_sne = []

    # Data flags are in binary
    # flags = [int(2 ** n) for n in range(0,10)]

    print('Searching for detections...')

    with mp.Pool() as pool:
        detected_sne = list(tqdm(
            pool.imap(partial(detect_sn, sn_info=sn_info, args=args), 
                sn_info.index, chunksize=10), 
            total=len(sn_info.index)
        ))

    # List of SNe with detections
    detected_sne = [band for sn in detected_sne for band in sn if len(sn) > 0]
    detected_sne = pd.DataFrame(detected_sne, columns=['Name', 'Band', 
            'Max Sigma', 'Background', 'Background Error', 'Systematic Error', 'Images'])
    output_csv(detected_sne, 'out/candidate_detections.csv', index=False)


def detect_sn(sn, sn_info, args):

    detected_sne = []
    
    short_name = Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_short.png')
    full_name = Path('lc_plots/' + sn.replace(':','_').replace(' ','_') + '_full.png')
    make_plot = (not full_name.is_file()) or args.overwrite

    if make_plot:
        # Initialize plot
        fig, ax = plt.subplots()
        xmin = xmax = 0

    for band in bands:
        # Import light curve
        try:
            lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
        except (FileNotFoundError, KeyError, IndexError, pd.errors.EmptyDataError):
            continue

        # Host background & systematic error
        # Detect if 3 points above 3 sigma, or 1 point above 5 sigma
        threshold = 3
        lc['sigma_above'] = lc['flux_hostsub'] / lc['flux_hostsub_err']
        lc.insert(0, 'name', np.array([sn] * len(lc.index)))
        lc.insert(1, 'band', np.array([band] * len(lc.index)))
        after = lc[lc['t_delta'] > DT_MIN]
        high_sigma = after[after['sigma_above'] >= 3]
        high_sigma_idx = ','.join(high_sigma.index.astype(str))
        entry = [sn, band, np.max(lc['sigma_above']), bg, bg_err, sys_err, high_sigma_idx]
        if len(high_sigma.index) >= 1:
            detected_sne.append(entry)
            threshold = 3
        # elif len(lc[lc['sigma_above'] >= 5].index) >= 1:
        #     detected_sne.append(entry)
        detections = lc[lc['sigma_above'] >= threshold].index

        if make_plot:
            # Plot data from this band
            fig, ax = plot_band(fig, ax, lc, band, bg, bg_err, args,
                    color=COLORS[band], detections=detections)
            # Figure out best x limits
            xmin = min((xmin, np.min(lc['t_delta'])))
            xmax = max((xmax, np.max(lc['t_delta'])))

    # Configure plot
    if make_plot:
        if xmax > 0 and len(lc.index) > 0:
            ax.set_xlabel('Observed time since discovery [days]')
            ax.set_xlim((xmin - 50, xmax + 50))
            ax.set_ylabel('Observed flux [erg s^-1 Å^-1 cm^-2]')
            plt.legend()
            fig.suptitle(sn)
            # Save full figure
            plt.savefig(full_name, bbox_inches='tight')
            short_range = lc[(lc['t_delta'] > DT_MIN) & (lc['t_delta'] < 1000)]
            # Save short figure
            if len(short_range.index) > 0:
                xlim = (short_range['t_delta'].iloc[0]-20, short_range['t_delta'].iloc[-1]+20)
                ax.set_xlim(xlim)
                plt.savefig(short_name, bbox_inches='tight')
        plt.close()

    return detected_sne


def plot_band(fig, ax, lc, band, bg, bg_err, args, marker='', color='', detections=[]):

    # Plot background average of epochs before discovery
    bg_alpha = 0.2
    plt.axhline(bg, 0, 1, color=color, alpha=0.5, linestyle='--', 
            linewidth=1)
    plt.axhspan(ymin=(bg - args.sigma * bg_err), ymax=(bg + args.sigma * bg_err), 
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