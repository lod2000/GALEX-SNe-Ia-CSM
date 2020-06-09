import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.time import Time

df = pd.read_csv(Path('out/fits_categories.csv'), index_col='File')
valued = df[pd.notna(df['Disc. Date'])]

fig, ax = plt.subplots()

# number of SNe per number of epochs for each category
epochs = valued['Total Epochs']
post = valued[valued['Category'] == 'post_disc']['Total Epochs']
both = valued[valued['Category'] == 'pre_post_disc']['Total Epochs']

b = 11
bins = np.logspace(0, np.log10(np.max(epochs)), b)
ax.hist(epochs, bins=bins, histtype='step', log=False, label='All SNe')
ax.hist(both, bins=bins, histtype='step', log=False, color='orange',
        label='SNe with epochs before \nand after discovery', linestyle='--')
ax.hist(post, bins=bins, histtype='step', log=False, color='green',
        label='SNe with multiple epochs \nafter discovery', linestyle=':')
ax.set_xlabel('Total # of epochs')
ax.set_xscale('log')
ax.xaxis.set_major_formatter(tkr.ScalarFormatter())
ax.set_ylabel('# of SNe')
#ax.set_ylim((0.5, 3e3))
#ax.yaxis.set_major_formatter(tkr.ScalarFormatter())
ax.legend()

plt.show()