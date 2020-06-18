import pandas as pd
import numpy as np
from pathlib import Path
from astropy.io.votable import parse
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy import units as u
from astroquery.ned import Ned
from astropy.io import ascii
from astropy.table import vstack, Table, MaskedColumn
from tqdm import tqdm
from time import sleep

QUERY_RADIUS = 1. # arcmin

def main():
    fits_info = pd.read_csv('out/fitsinfo.csv', index_col='Name')
    sample = pd.Series(fits_info.index.drop_duplicates())
    ref = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')

    ned_table = vstack([get_ned_table(sn, fits_info, ref) for sn in tqdm(sample)],
            join_type='outer', metadata_conflicts='silent')
    ned_table.remove_columns(['No.'])
    ned_table['Type'] = ned_table['Type'].astype(str)
    ned_table['Redshift Flag'] = ned_table['Redshift Flag'].astype(str)
    ned_table['Magnitude and Filter'] = ned_table['Magnitude and Filter'].astype(str)

    try:
        ascii.write(ned_table, 'out/ned_table.csv', format='csv', overwrite=True)
        ascii.write(ned_table, 'out/ned_table.tex', format='latex', overwrite=True)
    except PermissionError:
        ascii.write(ned_table, 'out/ned_table-tmp.csv', format='csv', overwrite=True)
        ascii.write(ned_table, 'out/ned_table-tmp.tex', format='latex', overwrite=True)


def get_ned_table(sn, fits_info, ref):
    hostname = ref.loc[sn, 'Host Name']
    if pd.isna(hostname):
        hostname = ''

    # First try a direct search by SN name
    ned_table = query_ned_name(sn, verb=0)
    ned_name = ned_table['Object Name'][0] if is_table(ned_table) else ''
    search_type = 'name'

    # Next try a search by host name
    if (not is_table(ned_table) or ned_table['Redshift'].mask[0]) and hostname != '':
        ned_table = query_ned_name(hostname)
        search_type = 'host'

    # Finally, search by location
    if not is_table(ned_table) or ned_table['Redshift'].mask[0]:
        ra, dec = fits_info.loc[sn, 'R.A.'], fits_info.loc[sn, 'Dec.']
        ned_table = query_ned_loc(ra, dec, radius=QUERY_RADIUS, verb=0)
        ned_sorted = ned_table[np.argsort(ned_table['Separation'])]
        ned_table = ned_sorted[0:1]
        # If location search turns up the original object, check again
        if ned_table['Object Name'][0] == ned_name and len(ned_sorted) > 1:
            ned_table = ned_sorted[1:2]
        search_type = 'loc'

    ned_table['SN Name'] = [sn]
    ned_table['Host Name'] = [hostname]
    ned_table['Search Type'] = [search_type]
    return ned_table


def is_table(ned_table):
    return (ned_table is not None and len(ned_table) > 0)


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
