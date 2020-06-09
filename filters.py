import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

fig, ax = plt.subplots()

os.chdir('filters')
filters = sorted(os.listdir(os.getcwd()))

for f in filters:
    inst = f.split('.')[0].replace('_',' ')
    name = f.split('.')[-2]
    array = np.loadtxt(f, delimiter=' ')
    ax.plot(array[:,0], array[:,1], label=' '.join([inst, name]))

ax.set_xlabel('Wavelength (Å)')
ax.set_xlim((1000, 4000))
ax.set_ylabel('Effective area (cm^2)')
ax.legend()

plt.show()
