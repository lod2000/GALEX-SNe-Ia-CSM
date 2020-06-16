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

def main():

    # Parse arguments for: fits file parent directory, SN reference info
    parser = argparse.ArgumentParser(description='Classify FITS images by relative timing to SN discovery date.')
    parser.add_argument('fits_dir', metavar='dir', type=Path, help='path to FITS data directory')
    parser.add_argument('-r', '--reference', type=Path, help='path to reference CSV with SN info', default='ref/osc-pre2014-v2-clean.csv')
    args = parser.parse_args()

    # Read clean reference csv (e.g. Open Supernova Catalog)
    ref = utils.import_osc(Path(args.reference))
    
    fits_dir = args.fits_dir
    fits_files = [f for f in fits_dir.glob('**/*.fits.gz')]

    # multiprocess classification
    with mp.Pool() as pool:
        stats = list(tqdm(
            pool.imap(partial(classify_fits, ref=ref), fits_files, chunksize=10), 
            total=len(fits_files)
        ))

    # Remove empty entries
    stats = list(filter(None, stats))

    df = pd.DataFrame(np.array(stats), columns=['Name', 'Disc. Date', 
            'R.A.', 'Dec.', 'Total Epochs', 'Epochs Pre-SN', 'Epochs Post-SN', 
            'First Epoch', 'Last Epoch'])

    try:
        df.to_csv('out/fits_stats.csv', index=False)
    # In case I forget to close the CSV first...
    except PermissionError:
        df.to_csv('out/fits_stats-tmp.csv', index=False)


def classify_fits(fits_file, ref):

    try:
        f = utils.Fits(fits_file)
        sn = utils.SN(fits_file, ref)
    except KeyError:
        # Skip if SN isn't found in reference info
        return []

    # Count number of GALEX epochs before / after discovery
    pre = len(f.tmeans[f.tmeans < sn.disc_date])
    post = len(f.tmeans[f.tmeans > sn.disc_date])

    return [sn.name, sn.disc_date.iso, 
            f.ra.to_string(unit=u.hour), f.dec.to_string(unit=u.degree), 
            f.epochs, pre, post, f.tmeans[0], f.tmeans[-1]]


if __name__ == '__main__':
    main()
