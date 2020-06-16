#!/usr/bin/env python

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from tqdm import tqdm
import utils
from pathlib import Path
import argparse
import multiprocessing as mp
from itertools import repeat
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

def main():

    # Parse arguments for: fits file parent directory, SN reference info
    parser = argparse.ArgumentParser(description='Classify FITS images by \
            relative timing to SN discovery date.')
    parser.add_argument('fits_dir', metavar='dir', type=Path, help='path to \
            FITS data directory')
    parser.add_argument('-r', '--reference', type=Path, help='path to reference\
            CSV with SN info', default='ref/osc-pre2014-v2-clean.csv')
    args = parser.parse_args()

    # Read clean reference csv (e.g. Open Supernova Catalog)
    ref = utils.import_osc(Path(args.reference))
    
    fits_info = compile_fits(args.fits_dir, ref)
    
    try:
        fits_info.to_csv('out/fitsinfo.csv', index=False)
    # In case I forget to close the CSV first...
    except PermissionError:
        fits_info.to_csv('out/fitsinfo-tmp.csv', index=False)

    plot_observations(fits_info)


def import_fits(fits_file, ref):
    """
    Imports FITS file
    Inputs:
        fits_file (Path): GALEX FITS file to import
        ref (DataFrame): SN reference info, e.g. from OSC
    Outputs:
        list of FITS file info, including number of observation epochs before 
        and after discovery
    """

    try:
        f = utils.Fits(fits_file)
        sn = utils.SN(fits_file, ref)
    except KeyError:
        # Skip if SN isn't found in reference info
        return []

    # Count number of GALEX epochs before / after discovery
    pre = len(f.tmeans[f.tmeans < sn.disc_date])
    post = len(f.tmeans[f.tmeans > sn.disc_date])

    return [sn.name, sn.disc_date.iso, f.band,
            f.ra.to_string(unit=u.hour), f.dec.to_string(unit=u.degree), 
            f.epochs, pre, post, f.tmeans[0], f.tmeans[-1]]


def compile_fits(fits_dir, ref):
    """
    Imports all FITS files and compiles info in single DataFrame
    Inputs:
        fits_dir (Path): parent directory of FITS files
        ref (DataFrame): SN reference info, e.g. from OSC
    Outputs:
        fits_info (DataFrame): table of info about all FITS files in fits_dir
    """

    fits_files = [f for f in fits_dir.glob('**/*.fits.gz')]

    with mp.Pool() as pool:
        stats = list(tqdm(
            pool.imap(partial(import_fits, ref=ref), fits_files, chunksize=10), 
            total=len(fits_files)
        ))

    # Remove empty entries
    stats = list(filter(None, stats))

    fits_info = pd.DataFrame(np.array(stats), columns=['Name', 'Disc. Date', 'Band',
            'R.A.', 'Dec.', 'Total Epochs', 'Epochs Pre-SN', 'Epochs Post-SN', 
            'First Epoch', 'Last Epoch'])

    return fits_info


def plot_observations(fits_info):
    """
    Plots histogram of the number of SNe with a given number of observations
    Inputs:
        fits_info (DataFrame): output from compile_fits
    """
    bands = ['FUV', 'NUV']

    fig, axes = plt.subplots(1,2, figsize=(12.0, 4.2))

    for i, band in enumerate(bands):
        df = fits_info[fits_info['Band'] == band]
        
        # number of SNe per number of epochs for each category
        epochs = df['Total Epochs']
        post = df[(df['Epochs Post-SN'] > 1) & (df['Epochs Pre-SN'] == 0)]['Total Epochs']
        both = df[(df['Epochs Post-SN'] > 0) & (df['Epochs Pre-SN'] > 0)]['Total Epochs']

        ax = axes[i]
        bins = np.logspace(0, np.log10(np.max(epochs)), 11)
        ax.hist(epochs, bins=bins, histtype='step', log=False, label='All SNe', align='left')
        ax.hist(both, bins=bins, histtype='step', log=False, color='orange', align='left',
                label='SNe with epochs before \nand after discovery', linestyle='--')
        ax.hist(post, bins=bins, histtype='step', log=False, color='green', align='left',
                label='SNe with multiple epochs \nafter discovery', linestyle=':')

        handles, labels = plt.gca().get_legend_handles_labels()
        ax.set_xlabel('Total # of epochs in ' + band)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(tkr.ScalarFormatter())
        ax.set_ylabel('# of SNe')

    fig.legend(handles, labels)

    plt.savefig(Path('out/observations.png'), bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()