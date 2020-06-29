#!/usr/bin/env python

import numpy as np
import pandas as pd
import requests
import time
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astroquery.ned import Ned
from astropy import units as u
from pathlib import Path
from time import sleep
import matplotlib.pyplot as plt

C = 3.e5 # km/s
H_0 = 70. # km/s/Mpc
OMEGA_M = 0.3
OMEGA_V = 0.7
WMAP = 4
CORR_Z = 1
QUERY_RADIUS = 5. # arcmin
BLOCK_SIZE = 10
NED_RESULTS_FILE = Path('out/scraped_table.csv')
NED_RESULTS_FILE_TMP = Path('out/scraped_table-tmp.csv')
BIB_FILE = Path('out/table_references.bib')
LATEX_TABLE_TEMPLATE = Path('ref/deluxetable_template.tex')
LATEX_TABLE_FILE = Path('out/table.tex')


def main():

    fits_info = pd.read_csv('out/fitsinfo.csv')
    fits_no_dup = fits_info.drop_duplicates('Name').set_index('Name')
    fits_info.set_index('Name', inplace=True)
    ref = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')

    prev = 'o'
    if NED_RESULTS_FILE.is_file():
        prev = input('Previous NED query results found. [K]eep/[c]ontinue/[o]verwrite? ')

    # Overwrite completely
    if prev == 'o':
        ned = pd.DataFrame()
        sne = np.array(fits_no_dup.index)
    # Continue from previous output
    elif prev == 'c':
        ned = pd.read_csv(NED_RESULTS_FILE, index_col='name', dtype={'z':float, 'h_dist':float})
        sne = np.array([row.name for i, row in fits_no_dup.iterrows() if row.name not in ned.index])
    # Keep previous output
    else:
        ned = pd.read_csv(NED_RESULTS_FILE, index_col='name', dtype={'z':float, 'h_dist':float})
        sne = np.array([])

    blocks = np.arange(0, len(sne), BLOCK_SIZE)
    for b in tqdm(blocks):
        sample = sne[b:min(b+BLOCK_SIZE, len(sne))]
        block = pd.concat([get_sn(sn, fits_no_dup, ref, verb=0) for sn in sample])
        ned = pd.concat([ned, block])
        try:
            ned.to_csv(NED_RESULTS_FILE)
        except PermissionError:
            ned.to_csv(NED_RESULTS_FILE_TMP)

    #plot_redshifts(ned)
    #print(get_catalogs(ned))
    to_latex(ned, fits_info)


def get_sn(sn, fits_info, ref, verb=0):
    """
    Retrieve SN info from NED. Uses astroquery to retrieve target names, then
    web scrapes to get target info.
    Inputs:
        sn (str): SN name
        fits_info (DataFrame): FITS file info, with duplicate entries removed
        ref (DataFrame): SN reference info, e.g. from OSC
        verb (int or bool, optional): vebrose output? default: False
    Output:
        sn_info (DataFrame): web-scraped info from NED
    """

    host = ref.loc[sn, 'Host Name']
    ra, dec = fits_info.loc[sn, 'R.A.'], fits_info.loc[sn, 'Dec.']
    if verb:
        print('\n\n%s, host %s, RA %s, Dec %s' % (sn, host, ra, dec))

    sn_info = pd.DataFrame([''], columns=['objname'])

    # Finally, try searching by location; if possible, use result with similar z
    # value to OSC
    nearest_query = query_loc(ra, dec, z=ref.loc[sn, 'z'], verb=verb)
    nearest_name = nearest_query['Object Name'].replace('+', '%2B')
    sn_info = scrape_overview(nearest_name, verb=verb)
    sn_info.loc[0,'sep'] = nearest_query['Separation']
    if pd.notna(sn_info.loc[0,'ra']):
        sn_info.loc[0,'offset'] = physical_offset(ra, dec, sn_info.loc[0,'ra'], 
                sn_info.loc[0,'dec'], sn_info.loc[0,'h_dist']) # kpc

    sn_info.loc[0,'name'] = sn
    sn_info.loc[0,'host'] = host
    sn_info.loc[0,'galex_ra'] = ra
    sn_info.loc[0,'galex_dec'] = dec

    return sn_info.set_index('name')


def query_name(objname, verb=0):
    """
    Query NED based on an object name (e.g., host galaxy name)
    Inputs:
        objname (str): name of object
        verb (int or bool, optional): vebrose output? default: False
    Outputs:
        ned_table: table of query results
    """

    if verb:
        print('\tsending query for %s...' % objname)
    try:
        results = Ned.query_object(objname)
        if verb:
            print('\tcomplete')
    except:
        if verb:
            print('Object name query failed for object: %s' % objname)
        results = None
    sleep(1)
    return results


def query_loc(ra, dec, radius=1., z=None, verb=0):
    """
    Query NED based on sky coordninates; return closest match with similar z
    Inputs:
        ra, dec (float): sky coords in HHhMMmSS.Ss str format
        radius (float, optional): query radius in arcmin, default=1
        verb (int or bool, optional): verbose output? Default: False
    Outputs:
        ned_table: astropy table of query results
    """

    coord = SkyCoord(ra, dec)
    # Astroquery search by location
    if verb:
        print('\tsending query...')
    ned_results = Ned.query_region(coord, radius=radius*u.arcmin)
    if verb:
        print('\tcomplete')
    # Sort results by separation from target coords
    ned_sorted = ned_results[np.argsort(ned_results['Separation'])]
    z_sorted = ned_sorted[ned_sorted['Redshift'].mask != True]
    # Choose closest result
    ned_table = ned_sorted[0]
    # If provided a z, search for result with similar z value
    if z:
        for object in z_sorted:
            if np.abs(object['Redshift'] - z) / z < 0.1:
                ned_table = object
                break
    sleep(1)
    if verb:
        print(ned_table)
    return ned_table


def scrape_overview(objname, verb=0):
    """
    Scrape NED by object name (e.g., host galaxy name or SN name)
    Inputs:
        objname (str): name of object
        verb (int or bool, optional): vebrose output? default: False
    Outputs:
        object_info (DataFrame): info from overview table in NED
    """

    # Get BeautifulSoup from URL
    url = 'https://ned.ipac.caltech.edu/byname?objname=%s&hconst=%s&omegam=%s&omegav=%s&wmap=%s&corr_z=%s' % (objname, H_0, OMEGA_M, OMEGA_V, WMAP, CORR_Z)
    if verb:
        print('\tscraping %s ...' % url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # mxpath labels in overview table
    main_mxpaths = dict(
        objname = 'NED_MainTable.main_col2',
        ra = 'NED_PositionDataTable.posn_col11',
        dec = 'NED_PositionDataTable.posn_col12',
        z = 'NED_BasicDataTable.basic_col4',
        z_err = 'NED_BasicDataTable.basic_col6',
        v_helio = 'NED_BasicDataTable.basic_col1',
        v_helio_err = 'NED_BasicDataTable.basic_col3',
        h_dist = 'NED_DerivedValuesTable.derived_col33',
        h_dist_err = 'NED_DerivedValuesTable.derived_col34',
        z_indep_dist = 'Redshift_IndependentDistances.ridist_col4[0]',
        type = 'NED_MainTable.main_col5',
        morph = "Classifications.class_col2[class_col1=='Galaxy Morphology']",
        a_v = 'NED_BasicDataTable.qlsize_col17',
        a_k = 'NED_BasicDataTable.qlsize_col27',
    )

    # class labels of references
    ref_classes = dict(
        posn_ref = 'ov_inside_coord_row',
        z_ref = 'ov_inside_redshift_row',
        morph_ref = 'ov_inside_classification_row',
        a_ref = 'ov_inside_prititle_row',
    )

    object_info = pd.DataFrame(columns=list(main_mxpaths.keys()))

    # Look for error messages
    err_msg = soup.find_all('div', class_='messages error')
    if len(err_msg) == 0: # if no error messages appear
        for key, mxpath in main_mxpaths.items():
            try:
                val = soup.find('span', mxpath=mxpath).get_text()
                if val == 'N/A':
                    val = np.nan
                object_info.loc[0, key] = val
            except AttributeError:
                continue
        for key, class_ in ref_classes.items():
            try:
                tr = soup.find('tr', class_=class_)
                ref = tr.find('a').get_text()
                object_info.loc[0, key] = ref
            except AttributeError:
                continue
    else:
        if verb:
            print('Object name scrape failed for object: %s' % objname)
        pass

    sleep(1)
    return object_info


def is_table(ned_table):
    """
    Returns whether the input NED table is real (at least one row) or not
    (None or 0 rows)
    """
    return (ned_table is not None and len(ned_table) > 0)


def physical_offset(ra1, dec1, ra2, dec2, h_dist):
    """
    Calculates physical offset, in kpc, between SN and host galaxy center
    Inputs:
        ra1, ra2, dec1, dec2 (str): coordinates of two objects in HHhMMmSS.Ss str format
        h_dist: Hubble distance from NED in Mpc
    """

    ra1, dec1, ra2, dec2 = Angle(ra1), Angle(dec1), Angle(ra2), Angle(dec2)
    diff = Angle(np.sqrt((ra1-ra2)**2 + (dec1-dec2)**2), u.rad)
    offset = h_dist * diff.value * 1000 # kpc
    return offset


def plot_redshifts(ned, bin_width=0.025):
    """
    Plots histogram of redshifts 
    """

    z = ned['z']
    z = z[pd.notna(z)].astype(float)
    bins = int((max(z) - min(z)) / bin_width)
    plt.hist(z, bins=bins, histtype='step')
    plt.xlabel('z')
    plt.xlim((0, max(z)))
    plt.ylabel('# of SNe')
    plt.savefig(Path('out/redshifts.png'), bbox_inches='tight', dpi=300)
    plt.xlim((0,0.5))
    plt.savefig(Path('out/redshifts_clipped.png'), bbox_inches='tight', dpi=300)
    plt.close()


def get_catalogs(ned):
    """
    Get catalog refcodes (denoted by trailing ':' in NED)
    """

    ned.dropna(subset=['z'], inplace=True)
    ref_cols = ['posn_ref', 'z_ref', 'morp_ref']
    catalogs = []
    for col in ref_cols:
        catalogs += list(ned[ned['posn_ref'].str.contains(':')]['posn_ref'])
    catalogs = list(dict.fromkeys(catalogs))
    return catalogs


def to_latex(ned, fits_info):
    """
    Outputs NED scrape output and important FITS info to LaTeX table
    """

    print('Preparing NED results...')
    # Cut SNe with z>=0.5 or unknown
    ned = ned[ned['z'] < 0.5]
    # Cut SNe with physical sep > 100 kpc
    ned = ned[ned['offset'] < 100]
    # Flag SNe with physical sep > 30 kpc
    ned.loc[ned['offset'] > 30, 'z_flag'] = 'large host offset'
    # Format coordinates, redshifts & distances
    ned['galex_coord'] = ned[['galex_ra', 'galex_dec']].agg(', '.join, axis=1)
    ned['z_str'] = ned['z'].round(6).astype('str').replace('0+$','',regex=True)
    ned['h_dist_str'] = ned['h_dist'].round().astype('str').replace('.','')
    # Sort by SN name
    ned.sort_index(inplace=True)
    ned.reset_index(inplace=True)
    # Add epoch counts
    # ned['epochs_total'] = np.sum(fits_info.loc[ned['name'], 'Total Epochs'])
    # print(ned['epochs_total'])

    # Get BibTeX entries and write bibfile
    overwrite = True
    if BIB_FILE.is_file():
        over_in = input('Previous bibliography detected. Overwrite? [y/N] ')
        overwrite = (over_in == 'y')

    if overwrite:
        print('Pulling BibTeX entries from ADS...')
        refs = list(ned['posn_ref']) + list(ned['z_ref']) + list(ned['morph_ref'])
        refs = list(dict.fromkeys(refs)) # remove duplicates
        bibcodes = {'bibcode':refs}
        with open('ads_token', 'r') as file:
            token = file.readline()
        ads_bibtex_url = 'https://api.adsabs.harvard.edu/v1/export/bibtex'
        r = requests.post(ads_bibtex_url, headers={'Authorization': 'Bearer ' + token}, data=bibcodes)
        bibtex = r.json()['export'].replace('A&A', 'AandA') # replace pesky ampersands
        with open(BIB_FILE, 'w') as file:
            file.write(bibtex)

    print('Writing to LaTeX table...')
    # Format reference bibcodes
    formatters = {'posn_ref':table_ref, 'z_ref':table_ref, 'morph_ref':table_ref}
    # Generate table with bare minimum data
    latex_table = ned.to_latex(na_rep='N/A', index=False, escape=False,
        columns=['name', 'galex_coord', 'z_str', 'h_dist_str', 'z_ref'],
        formatters=formatters
    )
    # Replace table header and footer with template
    # Edit this file if you need to change the number of columns or description
    with open(LATEX_TABLE_TEMPLATE, 'r') as file:
        dt_file = file.read()
        header = dt_file.split('===')[0]
        footer = dt_file.split('===')[1]
    latex_table = latex_table.split('\n')[4:-3]
    latex_table = header + '\n'.join(latex_table) + footer
    # Write table
    with open(LATEX_TABLE_FILE, 'w') as file:
        file.write(latex_table)


def table_ref(bibcode):
    """
    Formats reference bibcodes for LaTeX table
    """
    return '\citet{%s}' % bibcode.replace('A&A', 'AandA')


if __name__ == '__main__':
    main()