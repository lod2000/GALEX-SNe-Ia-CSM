# Clean up OSC CSV

import numpy as np
import pandas as pd
# import statistics
# from astropy.coordinates import SkyCoord

INPUT_FILE = 'OSC-pre2014-v2.csv'
OUTPUT_FILE = 'osc-pre2014-v2-clean.csv'

# Fix odd cases where R.A. or Dec. have 'AM' or 'PM' appended
# ang: string in format 'HH:MM:SS' or 'Deg:Min:Sec'
def fix_ang(ang):
    if ang.split(' ')[-1] == 'PM':
        ang = ang.split(' ')[0]
        ang = str(int(new[0:2])+12) + ang[2:]
    elif ang.split(' ')[-1] == 'AM':
        ang = ang.split(' ')[0]
    return ang


osc = pd.read_csv(INPUT_FILE, quotechar='"',skipinitialspace=True)

# Remove duplicate rows
osc = osc.drop_duplicates(['Name'], keep='first')

# Remove entries with incomplete / missing date
osc.replace('', float('NaN'), inplace=True)
osc.dropna(subset=['Disc. Date'], inplace=True)
osc = osc[osc['Disc. Date'].apply(lambda d: len(d) == 10)]
osc.reset_index(inplace=True, drop=True)

# Replace discovery date with ISO value
newdates = ['-'.join(str(date).split('/')) for date in osc['Disc. Date']]
osc['Disc. Date'] = pd.Series(newdates, dtype=str)
# Remove incomplete dates
osc = osc[osc['Disc. Date'].str.len() > 9]

# Replace maximum date with ISO value
newdates = []
for date in osc['Max Date']:
    if not pd.isna(date):
        newdates.append('-'.join(str(date).split('/')))
    else:
        newdates.append(date)
osc['Max Date'] = pd.Series(newdates, dtype=str)

# Replace multiple R.A. values with the most precise value
newra = []
for ra in osc['R.A.']:
    new = max(str(ra).split(','),key=len)
    new = fix_ang(new)
    newra.append(new)
osc['R.A.'] = newra

# Replace multiple DEC values with the most precise value
newdec = []
for dec in osc['Dec.']:
    newdec.append(fix_ang(max(str(dec).split(','),key=len)))
osc['Dec.'] = newdec

# Take the average of all z values
newz = []
for z in osc['z']:
    if pd.isna(z):
        newz.append(z)
    else:
        zvals = pd.Series(str(z).split(','))
        round_len = len(max(zvals,key=len)) # Number of decimals to round z average
        zvals = zvals.astype(float)
        newz.append(round(zvals.mean(), round_len-1))
osc['z'] = newz

osc.to_csv(OUTPUT_FILE, index=False)
