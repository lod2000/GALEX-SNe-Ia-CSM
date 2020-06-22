#!/usr/bin/env python

import numpy as np
import pandas as pd
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver import ActionChains
from requests_html import HTMLSession
import pyppdf.patch_pyppeteer
from tqdm import tqdm
from astropy.coordinates import SkyCoord
from astroquery.ned import Ned
from astropy import units as u
from pathlib import Path
from time import sleep

H_0 = 67.8 # km/sec/Mpc
OMEGA_M = 0.308
OMEGA_V = 0.692
WMAP = 4
CORR_Z = 1
QUERY_RADIUS = 1. # arcmin
NED_RESULTS_FILE = Path('out/scraped_table.csv')
NED_RESULTS_FILE_TMP = Path('out/scraped_table-tmp.csv')


def main():

    fits_info = pd.read_csv('out/fitsinfo.csv')
    posn_info = fits_info.drop_duplicates('Name').set_index('Name')
    sample = pd.Series(posn_info.index[:10])
    ref = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')

    gen_tab = True
    if NED_RESULTS_FILE.is_file():
        i = input('Previous NED query results found. Overwrite? [y/N] ')
        gen_tab = True if i == 'y' else False

    if gen_tab:
        # Scrape NED for all SNe in sample
        scraped = pd.concat([get_sn(sn, posn_info, ref, verb=0) for sn in tqdm(sample)],
                ignore_index=True)
        scraped.set_index('name')

        try:
            scraped.to_csv(NED_RESULTS_FILE, index=False)
        except PermissionError:
            scraped.to_csv(NED_RESULTS_FILE_TMP, index=False)
    else:
        ned = pd.read_csv(NED_RESULTS_FILE, index_col='name')


def get_sn(sn, fits_info, ref, verb=0):

    host = ref.loc[sn, 'Host Name']
    ra, dec = fits_info.loc[sn, 'R.A.'], fits_info.loc[sn, 'Dec.']
    if verb:
        print('\n\n%s, host %s, RA %s, Dec %s' % (sn, host, ra, dec))

    sn_info = pd.DataFrame()

    # First, try query by host name; if redshift data exists, scrape NED
    if pd.notna(host):
        host_query = query_name(host, verb=verb)
        if verb:
            print(host_query)
        if is_table(host_query) and not host_query['Redshift'].mask[0]:
            ned_host = host_query['Object Name'][0]
            sn_info = scrape_overview(ned_host, verb=verb)

    # Next, try a direct search by SN name
    if len(sn_info) == 0:
        sn_query = query_name(sn, verb=verb)
        if verb:
            print(sn_query)
        if is_table(sn_query) and not sn_query['Redshift'].mask[0]:
            sn_info = scrape_overview(sn, verb=verb)

    # Finally, try searching by location; if possible, use result with similar z
    # value to OSC
    if len(sn_info) == 0:
        nearest, sep = query_loc(ra, dec, z=ref.loc[sn, 'z'], verb=verb)
        sn_info = scrape_overview(nearest, verb=verb)
        sn_info.loc[0,'sep'] = sep

    sn_info.loc[0,'name'] = sn
    sn_info.loc[0,'host'] = host

    return sn_info


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
    Query NED based on sky coordninates
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
    '''
    # If location search turns up the original object, check again
    if ned_table['Object Name'] == objname and len(ned_sorted) > 1:
        ned_table = ned_sorted[1]
    '''
    sleep(1)
    if verb:
        print(ned_table)
    return ned_table['Object Name'], ned_table['Separation']


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
                object_info.loc[0, key] = val
            except AttributeError as e:
                raise e
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


if __name__ == '__main__':
    main()