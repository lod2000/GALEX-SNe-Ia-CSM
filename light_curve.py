from pathlib import Path

LC_DIR = Path('/mnt/d/GALEXdata_v10/LCs/')

class LightCurve:
    """GALEX light curve of a given supernova and band."""
    def __init__(self, fname, dir=LC_DIR, detrad_cut=0.55, fatal_flags=[], 
            manual_cuts=[]):
        """
        Imports a light curve file
        Inputs:
            fname: file name, should be similar to 'SN2007on-FUV.csv'
            dir: parent directory for light curve files
            detrad_cut: maximum detector radius value in degrees
            fatal_flags: additional fatal flags to eliminate, list of ints
            manual_cuts: indices to remove 'manually', list of ints
        """

        self.fname = Path(fname)
        self.sn, self.band = fname2sn(fname)

        # Import light curve CSV
        data = pd.read_csv(Path(dir) / self.fname)

        # Remove fatal flags
        self.fatal_flags = [1, 2, 4, 16, 64, 128, 512] + fatal_flags
        data = data[data['flags'] & reduce(or_, self.fatal_flags) == 0]

        # Cut sources outside detector radius
        detrad_cut_px = detrad_cut * 3600 / PLATE_SCALE
        data = data[data['detrad'] < detrad_cut_px]

        # Cut data with background much higher than average to eliminate washed-
        # out fields
        data.insert(29, 'bg_cps', lc['bg_counts'] / lc['exptime'])
        bg_median = np.median(lc['bg_cps'])
        high_bg = data[data['bg_cps'] > 3 * bg_median]

        # Cut unphysical data
        data = data[np.abs(data['flux_bgsub']) < 1] # extreme flux values
        data = data[data['bg_counts'] >= 0] # negative background counts

        # Manual cuts
        data = data[~data.index.isin(manual_cuts)]

        check_if_empty(lc, sn, band)

        # Cut data with background much higher than average (washed-out fields)
        # and output high backgrounds to file
        lc.insert(29, 'bg_cps', lc['bg_counts'] / lc['exptime'])
        bg_median = np.median(lc['bg_cps'])
        high_bg = lc[lc['bg_cps'] > 3 * bg_median]
        if len(high_bg.index) > 0:
            high_bg.insert(30, 'bg_cps_median', [bg_median] * len(high_bg.index))
            high_bg.insert(0, 'name', [sn] * len(high_bg.index))
            high_bg.insert(1, 'band', [band] * len(high_bg.index))
            if BG_FILE.is_file():
                high_bg = pd.read_csv(BG_FILE, index_col=0).append(high_bg)
                high_bg.drop_duplicates(inplace=True)
            output_csv(high_bg, BG_FILE, index=True)
            lc = lc[lc['bg_counts'] < 3 * bg_median]

        check_if_empty(lc, sn, band)


def fname2sn(fname):
    """Extract SN name and band from a file name."""

    fname = Path(fname)
    split = fname.stem.split('-')
    sn = '-'.join(split[:-1])
    band = split[-1]
    # Windows replaces : with _ in some file names
    if 'CSS' in sn or 'MLS' in sn:
        sn.replace('_', ':', 1)
    sn.replace('_', ' ')
    return sn, band


def sn2fname(sn, band, suffix='.csv'):
    # Converts SN name and GALEX band to a file name, e.g. for a light curve CSV

    fname = '-'.join((sn, band)) + suffix
    fname.replace(' ', '_')
    # Make Windows-friendly
    if (platform.system() == 'Windows') or ('Microsoft' in platform.release()):
        fname = fname.replace(':', '_')
    return Path(fname)

