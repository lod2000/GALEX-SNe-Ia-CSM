import numpy as np
import pandas as pd

ned = pd.read_csv('out/ned_table.csv', index_col='SN Name')
edd = pd.read_csv('ref/qualityDistances.csv', usecols=['name', 'pgc', 'DM', 'eDM'], index_col='name')
hyperleda = pd.read_csv('ref/hyperleda.info.cgi', sep='\s+\|\s', index_col='name', skipinitialspace=True)
hyperleda.replace(None, np.nan, inplace=True, regex='\s+\|')

comb = ned.copy()

for row in ned:
    hostname = row['Host Name']
    try:
        edd_entry = edd.loc[hostname]
        comb.loc[row.name, 'Distance Modulus'] = edd_entry['DM']
        comb.loc[row.name, 'Distance Modulus Error'] = edd_entry['eDM']
        comb.loc[row.name, 'Host PGC'] = edd_entry['pgc']
        hyperleda_entry = hyperleda.loc[hostname]
        comb.loc[row.name, 'Host Morphology'] = hyperleda_entry['type']
        comb.loc[row.name, 'Host Morphology Code'] = hyperleda_entry['t']
    except KeyError:
        continue

comb.to_csv('out/ned_edd.csv')