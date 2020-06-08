import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path('out/fits_categories.csv'), index_col='File')
epochs = df[pd.notna(df['Epochs Post-SN'])]['Total Epochs']
print(len(epochs))
post = df[pd.notna(df['Epochs Post-SN'])]['Epochs Post-SN']
print(len(post))

plt.hist([epochs, post], bins=20, 
        #range=(20, df['Total Epochs'].max()), 
        histtype='step', log=True)
plt.xlabel('# of epochs')
plt.ylabel('# of SNe')
plt.show()