import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from astroquery.ned import Ned
from tqdm import tqdm
from time import sleep

QUERY_RADIUS = 1. # arcmin

def main():
    fits_info = pd.read_csv('out/fitsinfo.csv', index_col='Name')
    sample = pd.Series(fits_info.index.drop_duplicates())
    sne = []
    objnames = []
    redshifts = []

    for sn in tqdm(sample):
        #print('\nQuerying Ned for: %s' % sn)
        ned_table = query_ned_name(sn, verb=0)
        if ned_table is None or len(ned_table) == 0:
            ra, dec = fits_info.loc[sn, 'R.A.'], fits_info.loc[sn, 'Dec.']
            ned_table = query_ned_loc(ra, dec, radius=QUERY_RADIUS, verb=0)
            best = np.argsort(ned_table['Separation'])[0:1]
            ned_table = ned_table[best]
            #print(ned_table)
            """
            print('Object name query failed.')
            ra, dec = fits_info.loc[sn, 'R.A.'], fits_info.loc[sn, 'Dec.']
            ned_table = query_ned_loc(ra, dec, radius=QUERY_RADIUS, verb=1)
            print('Location query results:')
            print(ned_table)
            best3 = np.argsort(ned_table['Separation'])[:3]
            print('Best matches:')
            print(ned_table[best3])
            """
        objnames.append(ned_table['Object Name'][0])
        redshifts.append(ned_table['Redshift'][0])

    df = pd.concat([sample, pd.Series(objnames), pd.Series(redshifts)], axis=1, keys=['Name', 'NED Object', 'Redshift'])
    try:
        df.to_csv('out/redshifts.csv', index=False)
    # In case I forget to close the CSV first...
    except PermissionError:
        df.to_csv('out/redshifts-tmp.csv', index=False)


def query_ned_loc(ra, dec, radius=1., verb=0):
    """
    Query NED based on sky coordninates
    Inputs:
        ra, dec (float): sky coords in HHhMMmSS.Ss str format
        radius (float, optional): query radius in arcmin, default=1
        verb (int or bool, optional): verbose output? Default: False
    Outputs:
        ned_table: astropy table of query results
    """

    #parse coords
    coord = SkyCoord(ra, dec)

    #send query
    if verb:
        print('\tsending query...')
    results = Ned.query_region(coord, radius=radius*u.arcmin)
    if verb:
        print('\tcomplete.')
    sleep(1)
    return results


def query_ned_name(objname, verb=0):
    """
    Query NEd based on an object name (e.g., host galaxy name)
    Inputs:
        objname (str): name of object
        verb (int or bool, optional): vebrose output? default: False
    Outputs:
        ned_table: table of query results
    """

    if verb:
        print('\tsending query...')
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


def get_ned_params(objname, verb=0):
    if verb:
        print('\tsending query...')
    try:
        redshifts = Ned.get_table(objname, 'redshifts')
        references = Ned.get_table(objname, 'references')
        if verb:
            print('\tcomplete')
    except:
        if verb:
            print('Object name query failed for object: %s' % objname)
        redshifts = None
        references = None
    return redshifts, references


#def query_ned_iau()


if __name__=='__main__':
    main()
