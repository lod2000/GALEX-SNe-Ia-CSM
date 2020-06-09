import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

fig, ax = plt.subplots()

filter_dir = Path('filters/')
filters = [f for f in filter_dir.glob('**/*.dat')]

for f in filters:
    name = f.name.split('.')[:-1]
    array = np.loadtxt(f, delimiter=' ')
    ax.plot(array[:,0], array[:,1])

plt.show()
