import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path('out/fits_categories.csv'), index_col='File')
df = df[pd.notna(df['Disc. Date'])]
bands = ['FUV', 'NUV']

fig, axes = plt.subplots(1,2, figsize=(12.0, 4.2))

for i, band in enumerate(bands):
    band_df = df[df.index.str.contains(band)]
    
    # number of SNe per number of epochs for each category
    epochs = band_df['Total Epochs']
    post = band_df[band_df['Category'] == 'post_disc']['Total Epochs']
    both = band_df[band_df['Category'] == 'pre_post_disc']['Total Epochs']

    ax = axes[i]
    b = 11
    bins = np.logspace(0, np.log10(np.max(epochs)), b)
    ax.hist(epochs, bins=bins, histtype='step', log=False, label='All SNe', align='left')
    ax.hist(both, bins=bins, histtype='step', log=False, color='orange', align='left',
            label='SNe with epochs before \nand after discovery', linestyle='--')
    ax.hist(post, bins=bins, histtype='step', log=False, color='green', align='left',
            label='SNe with multiple epochs \nafter discovery', linestyle=':')

    handles, labels = plt.gca().get_legend_handles_labels()
    ax.set_xlabel('Total # of epochs in ' + band)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(tkr.ScalarFormatter())
    ax.set_ylabel('# of SNe')

fig.legend(handles, labels)

plt.savefig(Path('figs/obs_epoch_hist.png'), bbox_inches='tight', dpi=300)
plt.show()
