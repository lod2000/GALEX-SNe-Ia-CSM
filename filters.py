import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

fig, ax = plt.subplots(figsize=(6.0, 4.2))

os.chdir('filters')
filters = ['GALEX.FUV.dat', 'GALEX.NUV.dat', 'Swift_UVOT.UVW2.dat',
        'Swift_UVOT.UVM2.dat', 'Swift_UVOT.UVW1.dat', 'HST_WFC3_UVIS2.F275W.dat']
colors = ['maroon', 'red', 'darkgreen', 'limegreen', 'chartreuse', 'indigo']
alphas = [1, 1, 0.6, 0.6, 0.6, 0.8]
styles = ['-', '-', '--', '--', '--', ':']

for i, f in enumerate(filters):
    inst = f.split('.')[0].replace('_',' ')
    name = f.split('.')[-2]
    array = np.loadtxt(f, delimiter=' ')
    freq = array[:,0]
    y = array[:,1]

    # convert effective area to throughput
    if inst == 'GALEX':
        y = y/(np.pi * 25**2) # objective diameter: 50 cm
    if inst == 'Swift UVOT':
        y = y/(np.pi * 15**2) # objective diameter: 30 cm

    ax.plot(freq, y, label=' '.join([inst, name]), linestyle=styles[i], 
            alpha=alphas[i], color=colors[i])

ax.set_xlabel('Wavelength (Å)')
ax.set_xlim((1000, 3500))
ax.set_ylabel('Transmission')
ax.legend()

plt.savefig(Path('../figs/filters.png'), bbox_inches='tight', dpi=300)
plt.show()
