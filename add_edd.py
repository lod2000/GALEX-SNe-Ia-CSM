import numpy as np
import pandas as pd
from pathlib import Path
import utils

sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')
if 'z_indep_dist' in sn_info.columns:
    sn_info.drop(columns=['z_indep_dist'], inplace=True)

cf3 = pd.read_csv('ref/qualityDistances.csv', sep='\s*,\s*', usecols=['name', 'pgc', 'DM', 'eDM'], index_col='name', skipinitialspace=True)
cf3.drop_duplicates(inplace=True)
cf3['dist_mpc'] = 10 ** (1/5 * (cf3['DM'] + 5)) * 1e-6
cf3['dist_err_mpc'] = cf3['dist_mpc'] * 1/5 * np.log(10) * cf3['eDM'] * 1e-6

hyperleda = pd.read_csv('ref/hyperleda.info.cgi', sep='\s+\|\s', index_col='name', skipinitialspace=True)
hyperleda.replace(None, np.nan, inplace=True, regex='\s+\|')
hyperleda.drop_duplicates(inplace=True)

for i, row in sn_info.iterrows():
    ned_hostname = row['host']
    ned_objname = row['objname']
    osc_hostname = row['osc_host']
    # Grab distance moduli from cosmic flows 3:
    if ned_hostname in cf3.index:
        hostname = ned_hostname
    elif ned_objname in cf3.index:
        hostname = ned_objname
    elif osc_hostname in cf3.index:
        hostname = osc_hostname
    else:
        hostname = np.nan

    if pd.notna(hostname):
        cf3_entry = cf3.loc[hostname]
        sn_info.loc[row.name, 'z_indep_dist'] = cf3_entry['dist_mpc']
        sn_info.loc[row.name, 'z_indep_dist_err'] = cf3_entry['dist_err_mpc']
        sn_info.loc[row.name, 'z_indep_ref'] = '2016AJ....152...50T'
        sn_info.loc[row.name, 'pgc'] = cf3_entry['pgc']
        sn_info.loc[row.name, 'pref_dist'] = cf3_entry['dist_mpc']
        sn_info.loc[row.name, 'pref_dist_err'] = cf3_entry['dist_err_mpc']
        sn_info.loc[row.name, 'pref_dist_ref'] = '2016AJ....152...50T'
    else:
        sn_info.loc[row.name, 'pref_dist'] = sn_info.loc[row.name, 'h_dist']
        sn_info.loc[row.name, 'pref_dist_err'] = sn_info.loc[row.name, 'h_dist_err']
        sn_info.loc[row.name, 'pref_dist_ref'] = sn_info.loc[row.name, 'z_ref']

    
    # Grab host morphology from hyperleda:
    if ned_hostname in hyperleda.index:
        hostname = ned_hostname
    elif ned_objname in hyperleda.index:
        hostname = ned_objname
    elif osc_hostname in hyperleda.index:
        hostname = osc_hostname
    else:
        hostname = np.nan

    if pd.notna(hostname):
        hyperleda_entry = hyperleda.loc[hostname]
        sn_info.loc[row.name, 'morph'] = hyperleda_entry['type']
        sn_info.loc[row.name, 'morph_ref'] = '2014A&A...570A..13M'
        sn_info.loc[row.name, 'pgc'] = hyperleda_entry['pgc']

utils.output_csv(sn_info, Path('ref/sn_info.csv'))