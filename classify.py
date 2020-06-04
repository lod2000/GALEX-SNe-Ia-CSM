#!/usr/bin/env python

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm
import utils
from pathlib import Path
import argparse
import multiprocessing as mp

def classify_fits(fits_file):
    f = utils.Fits(fits_file)
    exp_range = Time([f.header['EXPSTART'], f.header['EXPEND']], format='gps')

    category = ''
    # Check for multiple images; if not, sort into 'single' category
    if f.header['NAXIS'] == 2:
        category = 'single'
    # Some dates in the csv are missing or aren't specific enough
    elif pd.isna(f.sn.disc_date):
        category = 'unknown'
    elif exp_range.iso[1] < f.sn.disc_date:
        category = 'pre_disc'
    elif exp_range.iso[0] > f.sn.disc_date:
        category = 'post_disc'
    elif exp_range[0] < f.sn.disc_date and exp_range.iso[1] > f.sn.disc_date:
        category = 'pre_post_disc'
    # Note: might have some issues with discovery date not including time
    # information

    if category == 'single':
        epochs = 1
    else:
        epochs = f.header['NAXIS3']

    return [f.filename, category, epochs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify FITS images by relative\
            timing to SN discovery date.')
    parser.add_argument('fits_dir', metavar='dir', type=Path, 
            help='path to FITS data directory')
    args = parser.parse_args()

    # Read clean OSC csv
    osc = utils.import_osc(Path('ref/OSC-pre2014-expt-clean.csv'))

    #fits_dir = Path('/mnt/d/GALEX_SNeIa_REU/fits')
    #fits_dir = Path('/mnt/exthdd/GALEX_SNeIa_REU/fits')
    fits_dir = args.fits_dir
    fits_files = [f for f in fits_dir.glob('**/*.fits.gz')]

    #categories = []
    '''
    for fits_file in tqdm(fits_files):
        categories.append(classify_fits(fits_file))
    '''
    with mp.Pool(4) as pool:
        categories = list(tqdm(pool.imap(classify_fits, fits_files), 
                total=len(fits_files)))

    df = pd.DataFrame(np.array(categories), columns=['File', 'Category', 'Epochs'])
    try:
        df.to_csv('out/fits_categories.csv', index=False)
    except PermissionError:
        df.to_csv('out/fits_categories-tmp.csv', index=False)
