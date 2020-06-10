import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

fig, ax = plt.subplots(figsize=(6.0, 4.2))

os.chdir('filters')
filters = sorted(os.listdir(os.getcwd()))

for f in filters:
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

    # make GALEX curves more visible
    alpha = 1 if inst=='GALEX' else 0.8
    if 'Swift' in inst:
        style = '--'
    elif 'HST' in inst:
        style = ':'
    else:
        style = '-'

    ax.plot(freq, y, label=' '.join([inst, name]), linestyle=style, alpha=alpha)

ax.set_xlabel('Wavelength (Å)')
ax.set_xlim((1000, 3500))
#ax.set_yscale('log')
#ax.set_ylim((5e-3,0.14))
ax.set_ylabel('Transmission')
ax.legend()

plt.savefig(Path('../figs/filters.png'), bbox_inches='tight', dpi=300)
plt.show()
