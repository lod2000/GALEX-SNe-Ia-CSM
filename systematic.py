import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord

import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
from tqdm import tqdm

import phot
import utils


def plot_sys_error(fits_file):
    f = utils.Fits(fits_file)
    if f.header['NAXIS'] == 2:
        f.data = np.array([f.data])

    mags = []
    delta_mags = []
    tmeans = []
    star_ids = []

    for i, img in enumerate(tqdm(f.data)):
        # DAOPhot star locations
        stars = phot.find_stars(img, threshold=20)

        # GALEX magnitude conversion
        stars['ab_mag'] = utils.galex_ab_mag(stars['flux'], f.band)
        stars['delta_mag'] = utils.galex_delta_mag(stars['flux'], f.band, f.expts[i])

        # Convert pixels to sky coordinates
        sky_positions = [SkyCoord.from_pixel(s['xcentroid'], s['ycentroid'], \
                f.wcs) for s in stars]
        sky_positions = SkyCoord.from_pixel(stars['xcentroid'], stars['ycentroid'], f.wcs)
        stars['ra'] = sky_positions.ra
        stars['dec'] = sky_positions.dec

        mags.append(np.array(stars['ab_mag']))
        delta_mags.append(np.array(stars['delta_mag']))
        tmeans.append(np.array([f.tmeans[i]] * len(stars['ab_mag'])))
        star_ids.append(np.array(stars['id']))

    # Plot
    mags = np.concatenate(mags)
    delta_mags = np.concatenate(delta_mags)
    tmeans = np.concatenate(tmeans)
    star_ids = np.concatenate(star_ids)

    #id_cmap = mpl.colors.ListedColormap(['red', ''])
    plt.scatter(x=mags, y=delta_mags, s=0.5, c=tmeans, cmap=plt.get_cmap('Greys'))
    plt.xlabel('m_AB')
    plt.ylabel('delta m_AB')
    plt.show()


if __name__ == '__main__':
    single_img_fits = '/mnt/d/GALEX_SNeIa_REU/fits/ASASSN-13ch-FUV.fits.gz'
    many_img_fits = '/mnt/d/GALEX_SNeIa_REU/fits/PTF12hdb-NUV.fits.gz'
    plot_sys_error(Path(many_img_fits))