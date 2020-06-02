import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm
import utils
from pathlib import Path

## convert galex time to unix time: galex + 315964800

unknown = []
single = []
pre = []
post = []
both = []

# Read clean OSC csv
osc = utils.import_osc()

fits_dir = Path('/mnt/d/GALEX_SNeIa_REU/fits')
fits_files = [f for f in fits_dir.glob('**/*.fits.gz')]

for fits_file in tqdm(fits_files):
    # Pull SN name from fits file name
    sn_name = utils.fits2sn(fits_file, osc)
    # Observation band (N/FUV)
    band = fits_file.name.split('-')[-1].split('.')[0]
    sn_longname = [sn_name, band]
    #print(sn_longname)
    
    # Convert fits EXPSTART and EXPEND to ISO time
    # For this purpose we only care about the date
    with fits.open(fits_dir / fits_file) as hdu:
        exp = Time([hdu[0].header['EXPSTART'], hdu[0].header['EXPEND']],
            format='gps')
        exp.format = 'iso'
        naxis = hdu[0].header['NAXIS']
        #exp.out_subfmt='date'
        #print('EXPSTART: ' + str(exp[0]))
        #print('EXPEND: ' + str(exp[1]))
    
    disc_date = osc['Disc. Date'].loc[sn_name]
    
    try:
        # Check for multiple images; if not, sort into 'single' category
        if naxis == 2:
            single.append(sn_longname)
            #print('Single observation')
        # Some dates in the csv are missing or aren't specific enough
        elif pd.isna(disc_date) or len(disc_date.split('-')) < 3:
            unknown.append(sn_longname)
        else:
            # Discovery date in iso form
            disc_date = Time(disc_date, format='iso', out_subfmt='date')
            #print('DISCOVERY DATE: ' + str(disc_date))
            # Compare image dates to discovery date
            # Note: might have some issues with discovery date not including time
            # information
            if exp.iso[1] < disc_date:
                pre.append(sn_longname)
                #print('Multiple images before discovery')
            elif exp.iso[0] > disc_date:
                post.append(sn_longname)
                #print('Multiple images after discovery')
            elif exp.iso[0] < disc_date and exp.isot[1] > disc_date:
                both.append(sn_longname)
                #print('Data before and after discovery')
            #print('')
    except ValueError:
        print('ValueError exception, skipping the following entry:')
        print(disc_date)

pd.DataFrame(both).to_csv('pre_post_discovery.csv', index=False, header=False)
pd.DataFrame(post).to_csv('post_discovery.csv', index=False, header=False)
pd.DataFrame(pre).to_csv('pre_discovery.csv', index=False, header=False)
pd.DataFrame(single).to_csv('single_observation.csv', index=False, header=False)
pd.DataFrame(unknown).to_csv('unknown_discovery.csv', index=False, header=False)
