import numpy as np
import pandas as pd
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import re
from selenium import webdriver

H_0 = 67.8 # km/sec/Mpc
OMEGA_M = 0.308
OMEGA_V = 0.692


def main():

    main_mxpaths = dict(
        name = 'NED_NamesTable.name_col1[0]',
        ra = 'NED_PositionDataTable.posn_col11',
        dec = 'NED_PositionDataTable.posn_col12',
        # posn ref
        z = 'NED_BasicDataTable.basic_col4',
        z_err = 'NED_BasicDataTable.basic_col6',
        v_helio = 'NED_BasicDataTable.basic_col1',
        v_helio_err = 'NED_BasicDataTable.basic_col3',
        z_ref = 'NED_BasicDataTable.basic_col7',
        h_dist = 'NED_DerivedValuesTable.derived_col33',
        h_dist_err = 'NED_DerivedValuesTable.derived_col34',
        z_indep_dist = 'Redshift_IndependentDistances.ridist_col4[0]',
        type = 'NED_MainTable.main_col5',
        morph = "Classifications.class_col2[class_col1=='Galaxy Morphology']",
        morph_ref = "Classifications.class_col5[class_col1=='Galaxy Morphology']",
        a_v = 'NED_BasicDataTable.qlsize_col17',
        a_k = 'NED_BasicDataTable.qlsize_col27',
        # extinction ref
    )

    options = webdriver.FirefoxOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')
    options.add_argument('--headless')
    driver = webdriver.Firefox(executable_path='./geckodriver.exe', options=options)

    sn = 'SDSS-II SN 1740'
    url = 'https://ned.ipac.caltech.edu/byname?objname=%s&hconst=%s&omegam=%s&omegav=%s' % (sn, H_0, OMEGA_M, OMEGA_V)
    driver.get(url)
    page_source = driver.page_source
    response = requests.get(url)
    #soup = BeautifulSoup(response.text, "html.parser")
    soup = BeautifulSoup(page_source, 'lxml')

    # Redshift values from overview
    z_main = soup.find('span', mxpath=main_mxpaths['z']).get_text()
    z_main_err = soup.find('span', mxpath=main_mxpaths['z_err']).get_text()

    # Redshift values from Redshifts tab
    z_tab_href = soup.find('a', text=re.compile('Redshifts \([0-9]+\)'))['href']
    z_tab = soup.find('div', id=z_tab_href[1:])
    #preferred_z = z_tab.find('span', mxpath='NED_BasicDataTable.basic_col4').get_text()
    measured_z_table = z_tab.find('div', {'class': 'panel_collapsible_title'})
    #script = z_tab.find('script', {'type': 'text/javascript'})
    print(measured_z_table)

    # Redshift-independent distances from Distances tab
    dist_tab_href = soup.find('a', text=re.compile('Distances \([0-9]+\)'))['href']
    dist_tab = soup.find('div', id=dist_tab_href[1:])
    #print(dist_tab)


if __name__ == '__main__':
    main()