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
import matplotlib.pyplot as plt

QUERY_RADIUS = 1. # arcmin
NED_RESULTS_FILE = Path('out/ned_table.csv')
NED_RESULTS_FILE_TMP = Path('out/ned_table-tmp.csv')

def main():

    fits_info = pd.read_csv('out/fitsinfo.csv', index_col='Name')
    sample = pd.Series(fits_info.index.drop_duplicates())
    ref = pd.read_csv('ref/OSC-pre2014-v2-clean.csv', index_col='Name')

    gen_tab = True
    if NED_RESULTS_FILE.is_file():
        i = input('Previous NED query results found. Overwrite? [y/N] ')
        gen_tab = True if i == 'y' else False

    if gen_tab:
        # Query NED for all SNe in sample
        ned_table = vstack([get_ned_table(sn, fits_info, ref) for sn in tqdm(sample)],
                join_type='outer', metadata_conflicts='silent')
        ned_table.remove_columns(['No.'])
        ned_table['Type'] = ned_table['Type'].astype(str)
        ned_table['Redshift Flag'] = ned_table['Redshift Flag'].astype(str)
        ned_table['Magnitude and Filter'] = ned_table['Magnitude and Filter'].astype(str)
        ned = ned_table.to_pandas()

        try:
            ascii.write(ned_table, NED_RESULTS_FILE, format='csv', overwrite=True)
            ascii.write(ned_table, 'out/ned_table.tex', format='latex', overwrite=True)
        except PermissionError:
            ascii.write(ned_table, NED_RESULTS_FILE_TMP, format='csv', overwrite=True)
            ascii.write(ned_table, 'out/ned_table-tmp.tex', format='latex', overwrite=True)
    else:
        ned = pd.read_csv(NED_RESULTS_FILE, index_col='Name')

    #plot_redshifts(ned)


def plot_redshifts(ned, bin_width=0.05):
    z = ned['Redshift']
    z = z[pd.notna(z)]
    bins = int((max(z) - min(z)) / bin_width)
    plt.hist(z, bins=bins)
    plt.xlabel('z')
    plt.xlim((0, max(z)))
    plt.ylabel('# of SNe')
    plt.savefig(Path('out/redshifts.png'), bbox_inches='tight', dpi=300)
    plt.xlim((0,1))
    plt.savefig(Path('out/redshifts_clipped.png'), bbox_inches='tight', dpi=300)
    plt.close()


def get_ned_table(sn, fits_info, ref):
    """
    Searches for data about the specified supernova in NED. First it tries a
    direct name search; if that comes up empty, or doesn't yield a redshift
    value, then it tries to search by host galaxy name, and finally by proximity.
    The proximity search returns the object closest to the location of the SN
    that isn't the SN itself (if a direct name search didn't yield a redshift).
    Inputs:
        sn (str): name of the supernova to query
        fits_info (DataFrame): info about FITS files
        ref (DataFrame): info about SNe (e.g. from Open Supernova Catalog)
    Outputs:
        ned_table (Table): single-row astropy table with search results
    """

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

    ned_table['Name'] = [sn]
    ned_table['Host Name'] = [hostname]
    ned_table['Search Type'] = [search_type]
    return ned_table


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


def is_table(ned_table):
    """
    Returns whether the input NED table is real (at least one row) or not
    (None or 0 rows)
    """
    return (ned_table is not None and len(ned_table) > 0)


if __name__=='__main__':
    main()
