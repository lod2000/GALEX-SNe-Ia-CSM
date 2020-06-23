#!/usr/bin/env python

import numpy as np
import pandas as pd
import requests
#import urllib.request
import time
from bs4 import BeautifulSoup
import re
#from selenium import webdriver
#from selenium.webdriver import ActionChains
#from requests_html import HTMLSession
#import pyppdf.patch_pyppeteer
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astroquery.ned import Ned
from astropy import units as u
from pathlib import Path
from time import sleep

C = 3.e5
H_0 = 70. # km/sec/Mpc
OMEGA_M = 0.3
OMEGA_V = 0.7
WMAP = 4
CORR_Z = 1
QUERY_RADIUS = 5. # arcmin
BLOCK_SIZE = 10
NED_RESULTS_FILE = Path('out/scraped_table.csv')
NED_RESULTS_FILE_TMP = Path('out/scraped_table-tmp.csv')


def main():

    fits_info = pd.read_csv('out/fitsinfo.csv')
    posn_info = fits_info.drop_duplicates('Name').set_index('Name')
    sne = pd.Series(posn_info.index)
    ref = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')

    prev = 'o'
    if NED_RESULTS_FILE.is_file():
        prev = input('Previous NED query results found. [o]verwrite/[c]ontinue/[K]eep?')

    # Overwrite completely
    if prev == 'o':
        ned = pd.DataFrame()
        blocks = np.arange(0, len(sne), BLOCK_SIZE)
    # Continue from previous output
    elif prev == 'c':
        ned = pd.read_csv(NED_RESULTS_FILE, index_col='name')
        blocks = np.arange(len(ned), len(sne), BLOCK_SIZE)
    # Keep previous output
    else:
        ned = pd.read_csv(NED_RESULTS_FILE, index_col='name')
        blocks = np.array([])

    for b in tqdm(blocks):
        sample = sne[b:b+BLOCK_SIZE]
        block = pd.concat([get_sn(sn, posn_info, ref, verb=0) for sn in sample])
        ned = pd.concat([ned, block])
        try:
            ned.to_csv(NED_RESULTS_FILE)
        except PermissionError:
            ned.to_csv(NED_RESULTS_FILE_TMP)


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

    """
    # First, try query by host name; if redshift data exists, scrape NED
    if pd.notna(host):
        host_query = query_name(host, verb=verb)
        if verb:
            print(host_query)
        if is_table(host_query) and not host_query['Redshift'].mask[0]:
            ned_host = host_query['Object Name'][0].replace('+','%2B')
            sn_info = scrape_overview(ned_host, verb=verb)

    # Next, try a direct search by SN name
    if sn_info.loc[0,'objname'] == '' or pd.isna(sn_info.loc[0,'z']):
        sn_query = query_name(sn, verb=verb)
        if verb:
            print(sn_query)
        if is_table(sn_query) and not sn_query['Redshift'].mask[0]:
            sn_name = sn_query['Object Name'][0]
            sn_info = scrape_overview(sn, verb=verb)
    """

    # Finally, try searching by location; if possible, use result with similar z
    # value to OSC
    #if sn_info.loc[0,'objname'] == '' or pd.isna(sn_info.loc[0,'z']):
    nearest_query = query_loc(ra, dec, z=ref.loc[sn, 'z'], verb=verb)
    nearest_name = nearest_query['Object Name'].replace('+', '%2B')
    sn_info = scrape_overview(nearest_name, verb=verb)
    sn_info.loc[0,'sep'] = nearest_query['Separation']
    if pd.notna(sn_info.loc[0,'ra']):
        sn_info.loc[0,'offset'] = physical_offset(ra, dec, sn_info.loc[0,'ra'], 
                sn_info.loc[0,'dec'], sn_info.loc[0,'z']) # kpc

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

    return object_info


def is_table(ned_table):
    """
    Returns whether the input NED table is real (at least one row) or not
    (None or 0 rows)
    """
    return (ned_table is not None and len(ned_table) > 0)


def physical_offset(ra1, dec1, ra2, dec2, z):

    ra1, dec1, ra2, dec2 = Angle(ra1), Angle(dec1), Angle(ra2), Angle(dec2)
    diff = Angle(np.sqrt((ra1-ra2)**2 + (dec1-dec2)**2), u.rad)
    offset = hubble_distance(z) * diff.value * 1000 # kpc
    return offset


def hubble_distance(z):
    return C * float(z) / H_0 # Mpc


if __name__ == '__main__':
    main()