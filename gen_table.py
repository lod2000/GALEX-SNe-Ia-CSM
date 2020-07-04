import numpy as np
import pandas as pd
from pathlib import Path
import utils
import requests

SN_INFO_FILE = Path('ref/sn_info.csv')
BIB_FILE = Path('tex/table_references.bib')
CAT_FILE = Path('tex/catalog_codes.txt')
LATEX_TABLE_TEMPLATE = Path('tex/deluxetable_template.tex')
LATEX_TABLE_FILE = Path('tex/table.tex')
SHORT_TABLE_FILE = Path('tex/short_table.tex')


def main():
    """
    Outputs NED scrape output and important FITS info to LaTeX table
    """
    sn_info = pd.read_csv(SN_INFO_FILE, index_col='name')

    # Format coordinates, redshifts & distances
    sn_info['galex_coord'] = sn_info[['galex_ra', 'galex_dec']].agg(', '.join, axis=1)
    sn_info['z_str'] = sn_info['z'].round(6).astype('str').replace('0+$','',regex=True)
    sn_info['h_dist_str'] = sn_info['h_dist'].round(0).astype(int)
    # Add notes
    sn_info['notes'] = sn_info[['z_flag']].astype(str).replace('nan', 'N/A').agg('; '.join, axis=1)
    # Concat references
    sn_info['refs'] = sn_info[['z_ref', 'morph_ref']].astype('str').agg(';'.join, axis=1)

    # Get BibTeX entries and write bibfile
    overwrite = True
    if BIB_FILE.is_file():
        over_in = input('Previous bibliography detected. Overwrite? [y/N] ')
        overwrite = (over_in == 'y')

    if overwrite:
        print('Pulling BibTeX entries from ADS...')
        refs = list(sn_info['posn_ref']) + list(sn_info['z_ref']) + list(sn_info['morph_ref'])
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
    formatters = {'refs':table_ref}
    columns = ['name', 'disc_date', 'galex_coord', 'epochs_total', 
            'delta_t_first', 'delta_t_last', 'delta_t_next', 'z_str', 
            'h_dist_str', 'a_v', 'morph', 'refs']
    # Generate table
    sn_info.reset_index(inplace=True)
    latex_table = sn_info.to_latex(na_rep='N/A', index=False, escape=False,
        columns=columns, formatters=formatters
    )
    # Replace table header and footer with template
    # Edit this file if you need to change the number of columns or description
    with open(LATEX_TABLE_TEMPLATE, 'r') as file:
        dt_file = file.read()
        header = dt_file.split('===')[0]
        footer = dt_file.split('===')[1]
    latex_table = header + '\n'.join(latex_table.split('\n')[4:-3]) + footer
    # Write table
    with open(LATEX_TABLE_FILE, 'w') as file:
        file.write(latex_table)

    # Generate short table
    short = sn_info.iloc[0:19]
    short_table = short.to_latex(na_rep='N/A', index=False, escape=False,
        columns=columns, formatters=formatters
    )
    short_table = header + '\n'.join(short_table.split('\n')[4:-3]) + footer
    with open(SHORT_TABLE_FILE, 'w') as file:
        file.write(short_table)

    # Catalog bibcodes (NED has a weird format)
    get_catalogs(sn_info.copy())


def get_catalogs(sn_info):
    """
    Get catalog refcodes (denoted by trailing ':' in NED)
    """

    ref_cols = ['posn_ref', 'z_ref', 'morph_ref']
    sn_info[ref_cols] = sn_info[ref_cols].replace(np.nan, '')
    catalogs = []
    for col in ref_cols:
        catalogs += list(sn_info[sn_info[col].str.contains(':')][col])
    catalogs = list(dict.fromkeys(catalogs))
    catalog_str = '\n'.join(catalogs)
    with open(CAT_FILE, 'w') as file:
        file.write(catalog_str)
    return catalogs


def table_ref(bibcodes):
    """
    Formats reference bibcodes for LaTeX table
    Input:
        bibcodes (str): list of reference codes joined with ';'
    """
    bibcodes = bibcodes.replace(';nan', '').split(';')
    bibcodes = list(dict.fromkeys(bibcodes)) # remove duplicates
    return '\citet{%s}' % ','.join([b.replace('A&A', 'AandA') for b in bibcodes])


if __name__ == '__main__':
    main()