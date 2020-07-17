import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os
from utils import *

fig, ax = plt.subplots()
fig.set_tight_layout(True)

os.chdir('filters')
files = ['Swift_UVOT.UVW2.dat', 'Swift_UVOT.UVM2.dat', 'Swift_UVOT.UVW1.dat', 
         'HST_WFC3_UVIS2.F275W.dat', 'GALEX.FUV.dat', 'GALEX.NUV.dat']
aper_rad = {'GALEX': 25, 'Swift': 15, 'HST': None} # cm; None if already given transmission %
style = {'GALEX': '-', 'Swift': '--', 'HST': ':'}
alpha = {'GALEX': 1, 'Swift': 0.6, 'HST': 0.8}

for i, f in enumerate(files):
    telescope = f.split('.')[0].split('_')[0]
    band = f.split('.')[1]
    label = ' '.join((telescope, band))
    array = np.loadtxt(f, delimiter=' ')
    freq = array[:,0]
    y = array[:,1]

    # convert effective area to transmission
    if aper_rad[telescope] != None:
        y = y/(np.pi * aper_rad[telescope]**2)

    ax.plot(freq, y*100, label=label, linestyle=style[telescope], alpha=alpha[telescope],
            color=colors[band])

ax.set_xlabel('Wavelength [Å]')
ax.set_xlim((1200, 3300))
ax.set_ylim((-0.5, None))
ax.set_ylabel('Transmission [%]')
ax.legend()

plt.savefig(Path('../figs/filters.png'), bbox_inches='tight', dpi=300)
plt.show()
