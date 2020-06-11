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

def classify_fits(fits_file):
    try:
        f = utils.Fits(fits_file)
    except KeyError:
        return [fits_file.name] + [np.nan] * 7
    exp_range = Time([f.header['EXPSTART'], f.header['EXPEND']], format='gps')

    # Categorize based on discovery and image dates
    category = ''
    # Check for multiple images; if not, sort into 'single' category
    if f.header['NAXIS'] == 2:
        category = 'single'
        epochs = 1
    elif exp_range.iso[1] < f.sn.disc_date:
        category = 'pre_disc'
    elif exp_range.iso[0] > f.sn.disc_date:
        category = 'post_disc'
    elif exp_range[0] < f.sn.disc_date and exp_range.iso[1] > f.sn.disc_date:
        category = 'pre_post_disc'
    # Note: might have some issues with discovery date not including time
    # information

    # Total epochs
    if not category == 'single':
        epochs = f.header['NAXIS3']
    # Epochs before and after explosion
    if len(f.tmeans) > 0 and not pd.isna(f.sn.disc_date):
        pre = len(f.tmeans[f.tmeans < f.sn.disc_date.gps])
        post = len(f.tmeans[f.tmeans > f.sn.disc_date.gps])
    else:
        pre = post = np.nan

    disc_date = f.sn.disc_date
    if pd.notna(disc_date):
        disc_date.out_subfmt = 'date'
        disc_date = disc_date.iso
    return [f.filename, disc_date, f.ra.to_string(unit=u.hour), 
            f.dec.to_string(unit=u.degree), category, epochs, pre, post]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify FITS images by relative\
            timing to SN discovery date.')
    parser.add_argument('fits_dir', metavar='dir', type=Path, 
            help='path to FITS data directory')
    args = parser.parse_args()

    # Read clean OSC csv
    osc = utils.import_osc(Path('ref/OSC-pre2014-v2-clean.csv'))
    
    fits_dir = args.fits_dir
    fits_files = [f for f in fits_dir.glob('**/*.fits.gz')]

    # multiprocess classification
    with mp.Pool() as pool:
        categories = list(tqdm(pool.imap(classify_fits, fits_files), 
                total=len(fits_files)))

    df = pd.DataFrame(np.array(categories), columns=['File', 'Disc. Date', 'R.A.',
            'Dec.', 'Category', 'Total Epochs', 'Epochs Pre-SN', 'Epochs Post-SN'])
    skipped = df[df['Disc. Date'].isna()]
    print(str(len(df.index)) + ' files skipped becausue of missing entry in OSC database.')
    df = df.dropna()

    try:
        df.to_csv('out/fits_categories.csv', index=False)
    # In case I forget to close the CSV first...
    except PermissionError:
        df.to_csv('out/fits_categories-tmp.csv', index=False)
