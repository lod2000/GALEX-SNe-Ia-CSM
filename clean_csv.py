# Clean up OSC CSV

import numpy as np
import pandas as pd
import statistics

csvfile = 'OSC-pre2014-expt.csv'
osc = pd.read_csv(csvfile, quotechar='"',skipinitialspace=True)
print(osc)

# Replace discovery date with ISO value
newdates = []
for date in osc['Disc. Date']:
    if not pd.isna(date):
        newdates.append('-'.join(str(date).split('/')))
    else:
        newdates.append(date)
osc['Disc. Date'] = pd.Series(newdates, dtype=str)

# Replace multiple R.A. values with the most precise value
newra = []
for ra in osc['R.A.']:
    newra.append(max(str(ra).split(','),key=len))
osc['R.A.'] = newra

# Replace multiple DEC values with the most precise value
newdec = []
for dec in osc['Dec.']:
    newdec.append(max(str(dec).split(','),key=len))
osc['Dec.'] = newdec

# Take the average of all z values
newz = []
for z in osc['z']:
    if pd.isna(z):
        newz.append(np.nan)
    else:
        zvals = pd.Series(str(z).split(','))
        round_len = len(max(zvals,key=len)) # Number of decimals to round z average
        zvals = zvals.astype(float)
        newz.append(round(zvals.mean(), round_len-1))
osc['z'] = newz

print(osc)

osc.to_csv('OSC-pre2014-expt-clean.csv', index=False)
