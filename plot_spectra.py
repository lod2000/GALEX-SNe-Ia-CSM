import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

files = [Path('SN2005gj_spec_49.csv'), Path('PTF11kx_spec_55.csv'), Path('SN2011fe_spec_45.csv')]
scale = [3e15, 5e15, 1e13]
redshifts = [0.062, 0.0467, 0.0008]
offsets = [1.5, 0.9, 0]

fig, ax = plt.subplots()
fig.set_tight_layout(True)

for file, s, z, o in zip(files, scale, redshifts, offsets):
    spec = pd.read_csv(Path('external') / file)
    name = file.name.split('_')[0]
    epoch = file.name.split('_')[-1].split('.')[0]
    ax.plot(spec['wavelength'] * 1/(1+z), spec['flux'] * s + o, 
            label='%s (+%s d)' % (name, epoch))

ax.set_xlabel('Rest wavelength [Å]')
ax.set_xlim((3300, 9000))
ax.set_ylabel('Flux [scaled, offset]')
ax.set_ylim((0, 4))

plt.legend()

plt.show()
