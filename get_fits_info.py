#!/usr/bin/env python

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from tqdm import tqdm
from pathlib import Path
import argparse
import warnings

import multiprocessing as mp
from itertools import repeat
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

import utils

FITS_INFO_FILE = Path('ref/fits_info.csv')
SN_INFO_FILE = 'ref/sn_info.csv'
STATS_FILE = 'out/quick_stats.txt'


def main():

    # Parse arguments for: fits file parent directory, SN reference info
    parser = argparse.ArgumentParser(description='Classify FITS images by \
            relative timing to SN discovery date.')
    parser.add_argument('fits_dir', metavar='dir', type=Path, help='path to \
            FITS data directory')
    args = parser.parse_args()

    # Suppress Astropy warnings about "dubious years", etc.
    warnings.simplefilter('ignore', category=AstropyWarning)

    # Read Open Supernova Catalog
    osc = utils.import_osc()
    
    # Option to overwrite or keep
    overwrite = True
    if FITS_INFO_FILE.is_file():
        over_in = input('Previous FITS info file detected. Overwrite? [y/N] ')
        overwrite = (over_in == 'y')

    if overwrite:
        # Get all FITS file paths
        fits_files = utils.get_fits_files(args.fits_dir, osc)
        # Import all FITS files
        fits_info = compile_fits(fits_files, osc)
        utils.output_csv(fits_info, FITS_INFO_FILE, index=False)
    else:
        fits_info = pd.read_csv(FITS_INFO_FILE)

    # Select only those with before+after observations
    final_sample = get_final_sample(fits_info) 

    # Output compressed CSV without SN name duplicates
    sn_info = compress_duplicates(final_sample.copy())
    utils.output_csv(sn_info, SN_INFO_FILE)

    # Plot histogram of observations
    plot_observations(fits_info)

    # Write a few statistics about FITS files
    write_quick_stats(fits_info, final_sample, sn_info, osc, STATS_FILE)


def import_fits(fits_file, osc):
    """
    Imports FITS file
    Inputs:
        fits_file (Path): GALEX FITS file to import
        osc (DataFrame): Open Supernova Catalog reference info
    Outputs:
        list of FITS file info, including number of observation epochs before 
        and after discovery
    """

    try:
        f = utils.Fits(fits_file)
        sn_name = utils.fits2sn(fits_file, osc)
        sn = utils.SN(sn_name, osc)
    except KeyError:
        # Skip if SN isn't found in reference info, or FITS file is incomplete
        return None

    # Count number of GALEX epochs before / after discovery
    pre = len(f.tmeans[f.tmeans < sn.disc_date])
    post = len(f.tmeans[f.tmeans > sn.disc_date])
    if post > 0:
        diffs = f.tmeans.mjd - sn.disc_date.mjd
        min_post = int(np.round(np.min(diffs[diffs >= 0]))) # earliest post-disc observation
    else:
        min_post = np.nan

    return [sn.name, sn.disc_date.iso, f.band,
            f.ra.to_string(unit=u.hour), f.dec.to_string(unit=u.degree), 
            f.epochs, pre, post, int(sn.disc_date.mjd - f.tmeans[0].mjd), 
            int(f.tmeans[-1].mjd - sn.disc_date.mjd), min_post, f.filename]


def compile_fits(fits_files, osc):
    """
    Imports all FITS files and compiles info in single DataFrame
    Inputs:
        fits_files (list): list of paths of FITS files
        osc (DataFrame): Open Supernova Catalog reference info
    Outputs:
        fits_info (DataFrame): table of info about all FITS files in fits_dir
    """

    print('\nCompiling FITS info...')

    with mp.Pool() as pool:
        stats = list(tqdm(
            pool.imap(partial(import_fits, osc=osc), fits_files, chunksize=10), 
            total=len(fits_files)
        ))

    # Remove empty entries
    stats = list(filter(None, stats))

    fits_info = pd.DataFrame(np.array(stats), columns=['Name', 'Disc. Date', 'Band',
            'R.A.', 'Dec.', 'Total Epochs', 'Epochs Pre-SN', 'Epochs Post-SN', 
            'First Epoch', 'Last Epoch', 'Next Epoch', 'File'])
    fits_info = fits_info.astype({'Total Epochs':int, 'Epochs Pre-SN':int, 'Epochs Post-SN':int})

    return fits_info


def get_post_obs(fits_info):
    """
    Returns FITS files with multiple post-discovery observations
    Input:
        fits_info (DataFrame): output from compile_fits
    """

    post = fits_info['Epochs Post-SN']
    pre = fits_info['Epochs Pre-SN']
    return fits_info[(post > 1) & (pre == 0)].reset_index(drop=True)


def get_pre_post_obs(fits_info):
    """
    Returns FITS files with both pre- and post-discovery observations
    Input:
        fits_info (DataFrame): output from compile_fits
    """

    post = fits_info['Epochs Post-SN']
    pre = fits_info['Epochs Pre-SN']
    return fits_info[(post > 0) & (pre > 0)].reset_index(drop=True)


def get_final_sample(fits_info):
    """
    Strips out FITS files with only a single observation, or with only 
    pre-discovery observations
    Input:
        fits_info (DataFrame): output from compile_fits
    Output:
        sample (DataFrame): stripped-down final sample
    """

    # post = get_post_obs(fits_info)
    both = get_pre_post_obs(fits_info)
    # return post.append(both).sort_values(by=['Name', 'Band']).reset_index(drop=True)
    sample = both.sort_values(by=['Name', 'Band']).set_index('Name', drop=True)
    return sample


def compress_duplicates(fits_info):
    """
    Compressses fits_info down to one entry per SN (removing band-specific
    information). Observation epochs are summed, and first/last/next epochs are
    maximized.
    Input:
        fits_info (DataFrame): FITS file-specific information
    Output:
        sn_info (DataFrame): SN-specific information
    """

    duplicated = fits_info.groupby(['R.A.', 'Dec.'])
    sn_info = pd.DataFrame([], index=pd.Series(fits_info.index, name='name'))
    sn_info[['disc_date', 'galex_ra', 'galex_dec']] = fits_info[['Disc. Date', 'R.A.', 'Dec.']].copy()
    sn_info['epochs_total'] = duplicated['Total Epochs'].transform('sum')
    sn_info['epochs_pre'] = duplicated['Epochs Pre-SN'].transform('sum')
    sn_info['epochs_post'] = duplicated['Epochs Post-SN'].transform('sum')
    sn_info['delta_t_first'] = duplicated['First Epoch'].transform('max')
    sn_info['delta_t_last'] = duplicated['Last Epoch'].transform('max')
    sn_info['delta_t_next'] = duplicated['Next Epoch'].transform('min')
    sn_info = sn_info.drop_duplicates()
    return sn_info


def write_quick_stats(fits_info, final_sample, sn_info, osc, file):
    """
    Writes quick statistics about sample to text file
    Input:
        fits_info (DataFrame): output from compile_fits
        final_sample (DataFrame): output from get_final_sample
        sn_info (DataFrame): output from compress_duplicates
        osc (DataFrame): Open Supernova Catalog reference info
        file (Path or str): output file
    """

    print('Writing quick stats...')
    sne = fits_info.loc[fits_info.index.drop_duplicates()]
    post = get_post_obs(fits_info).drop_duplicates(['Name'])
    both = get_pre_post_obs(fits_info).drop_duplicates(['Name'])
    # final_sne = final_sample.drop_duplicates(['Name'])
    fuv = final_sample[final_sample['Band'] == 'FUV']
    nuv = final_sample[final_sample['Band'] == 'NUV']
    with open(file, 'w') as f:
        f.write('Quick stats:\n')
        f.write('\tnumber of reference SNe: %s\n' % len(osc))
        f.write('\tnumber of SNe with GALEX data: %s\n' % len(sne))
        f.write('\tnumber of SNe with multiple observations after discovery: %s\n' % len(post))
        f.write('\tnumber of SNe with observations before and after discovery: %s\n' % len(both))
        f.write('\tfinal sample size: %s\n' % len(sn_info.index))
        f.write('\tnumber of final SNe with FUV observations: %s\n' % len(fuv))
        f.write('\tnumber of final SNe with NUV observations: %s\n' % len(nuv))


def plot_observations(fits_info):
    """
    Plots histogram of the number of SNe with a given number of observations
    Inputs:
        fits_info (DataFrame): output from compile_fits
    """

    print('\nPlotting histogram of observation frequency...')
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
