import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.time import Time
import matplotlib.markers as mmarkers
import matplotlib.colors as mcolors
import utils
from lc_utils import *
from tqdm import tqdm


def main():

    sigma = 3

    sn_info = pd.read_csv('ref/sn_info.csv', index_col='name')

    # Initialize output DataFrames
    # flagged_points = []    
    # total_points = 0
    detections = []
    bands = ['FUV', 'NUV']
    # bg_list = []
    # bg_loc = []
    # exptimes = np.array([])

    # Data flags are in binary
    flags = [int(2 ** n) for n in range(0,10)]

    print('Searching for detections...')
    # print('Plotting light curves...')
    for sn in tqdm(sn_info.index):

        for band in bands:

            try:
                lc = full_import(sn, band, sn_info)
            except FileNotFoundError:
                continue

            bg, bg_err, sys_err = get_background(lc, band, 'flux_bgsub')
            lc['sigma_above'] = lc['flux_hostsub'] / lc['flux_hostsub_err']
            lc['bg_sigma_above'] = lc['flux_hostsub'] / bg_err
            lc['point_sigma_above'] = lc['flux_hostsub'] / lc['flux_bgsub_err_total']

            # Detect if 3 points above 3 sigma, or 1 point above 5 sigma
            if len(lc[lc['sigma_above'] > 3].index) >= 3 or len(lc[lc['sigma_above'] > 5].index) >= 1:
                detections.append([sn, band, np.max(lc['sigma_above'])])

    detections = pd.DataFrame(detections, columns=['Name', 'Band', 'Max Sigma'])
    utils.output_csv(detections, 'out/detections.csv', index=False)


def short_plot():
    pass


def full_plot():
    pass


if __name__ == '__main__':
    main()