import numpy as np
import pandas as pd
from pathlib import Path

data = []
for file in ['ASASSN_I.txt', 'ASASSN_II.txt', 'ASASSN_III.txt', 'ASASSN_IV.txt']:
    if file == 'ASASSN_IV.txt':
        cols = [0, 1, 10]
    else:
        cols = [0, 1, 9]
    file_path = Path('external/%s' % file)
    df = pd.read_csv(file_path, sep='\s+', index_col=0, na_values='---', 
            skiprows=31, usecols=cols, names=['SNNAME', 'IAUName', 'Type'])
    data.append(df)

data = pd.concat(data)
print(data)

all_Ia = data[data['Type'].str.contains('Ia')]
Ia_CSM = data[(data['Type'] == 'Ia+CSM') | (data['Type'] == 'Ia-CSM')]
Ia_91T = data[data['Type'] == 'Ia-91T']

print('Ia-CSM: %s' % len(Ia_CSM.index))
print('Ia-91T: %s' % len(Ia_91T.index))
print('All Ia: %s' % len(all_Ia.index))
