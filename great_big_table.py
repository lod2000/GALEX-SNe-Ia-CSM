import pandas as pd
import numpy as np
from pathlib import Path


H0 = 70e3 # m/s/Mpc
c = 3e8 # m/s


def hubble_relation(z):
    return c*z/H0 # Mpc


fits_dir = Path('sample/')
fits_files = [f for f in fits_dir.glob('**/*.fits.gz')]

for fits_file in fits_files:
    f = utils.Fits(fits_file)
    #tmax = f.sn.max_date
    tmax = f.sn.disc_date
    tfirst = f.header['EXPSTART']
    tlast = f.header['EXPEND']
    dtfirst = (tmax - tfirst).jd
    dtlast = (tlast - tmax).jd

    dist = hubble_relation(f.sn.z)

df = pd.read_csv(Path('out/fits_categories.csv'), index_col='File')

df.to_latex(Path('out/table.tex'))