import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import os

fig, ax = plt.subplots()

os.chdir('filters')
filters = ['GALEX.FUV.dat', 'GALEX.NUV.dat', 'Swift_UVOT.UVW2.dat',
        'Swift_UVOT.UVM2.dat', 'Swift_UVOT.UVW1.dat', 'HST_WFC3_UVIS2.F275W.dat']
labels = ['GALEX FUV', 'GALEX NUV', 'Swift UVW2', 'Swift UVM2', 'Swift UVW1', 'HST F275W']
colors = ['maroon', 'red', 'darkgreen', 'limegreen', 'chartreuse', 'indigo']
alphas = [1, 1, 0.6, 0.6, 0.6, 0.8]
styles = ['-', '-', '--', '--', '--', ':']
conversion_radii = [25, 25, 15, 15, 15, None] # cm; None if already given transmission %

for i, f in enumerate(filters):
    inst = f.split('.')[0].replace('_',' ')
    name = f.split('.')[-2]
    label = labels[i]
    array = np.loadtxt(f, delimiter=' ')
    freq = array[:,0]
    y = array[:,1]

    # convert effective area to transmission
    if conversion_radii[i] != None:
        y = y/(np.pi * conversion_radii[i]**2)

    ax.plot(freq, y*100, label=label, linestyle=styles[i], 
            alpha=alphas[i], color=colors[i])

ax.set_xlabel('Wavelength [Å]')
ax.set_xlim((1200, 3300))
ax.set_ylim((0, None))
ax.set_ylabel('Transmission [%]')
ax.legend()

plt.savefig(Path('../figs/filters.png'), bbox_inches='tight', dpi=300)
plt.show()
