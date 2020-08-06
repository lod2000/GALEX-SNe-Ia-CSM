from tqdm import tqdm

from utils import *
from CSMmodel import CSMmodel

DT = 100 # start time iteration (days)
DECAY_RATE = 0.3
SCALE_FACTORS = [0.1, 0.5, 1, 2]
WIDTH = 200 # plateau width (days)
SIGMA = 3 # significance of data above which it counts as a detection
BINS = [0, 100, 500, 2500] # bin limits for binomial statistics on constraints (days)

def main():

    sn_info = pd.read_csv(Path('ref/sn_info.csv'), index_col='name')

    # For testing
    sn = 'SN2007on'
    band = 'NUV'
    tstart = 300
    twidth = 200

    rec = inject_recover(sn, band, sn_info, tstart, twidth)

    print(rec)
    """
    z = sn_info.loc[sn, 'z']
    lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
    post_lc = lc[lc['t_delta_rest'] > 0] # post-discovery light curve
    nondet = post_lc[post_lc['sigma'] < SIGMA] # nondetections only

    scale = SCALE_FACTORS[1]

    tstart = 0
    tmax = np.max(lc['t_delta_rest'])

    while tstart < tmax:
        # Initialize CSM model
        model = CSMmodel(tstart, WIDTH, DECAY_RATE, scale=scale)
        # Calculate luminosity at observation epochs
        model_luminosity = model(nondet['t_delta_rest'], z)[band]
        # If model luminosity is greater than 1-sigma nondetection limit,
        # treat it as a constraint
        diff = nondet['luminosity_hostsub_err'] - model_luminosity
        constraint_idx = diff[diff < 0].index
        # List of observation epochs at which GALEX nondetection(s) rule out
        # this particular CSM model
        constraint_epochs = nondet.loc[constraint_idx, 't_delta_rest'].to_list()
        print(constraint_epochs)
        tstart += DT
    """

    """
    for sn in tqdm(sn_info.index):
        for band in ['FUV', 'NUV']:
            lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
    """


def inject_recover(sn, band, sn_info, tstart, twidth, scale=1):
    """
    Performs injection and recovery for given SN and model parameters
    """

    z = sn_info.loc[sn, 'z']
    lc, bg, bg_err, sys_err = full_import(sn, band, sn_info)
    lc = inject_model(lc, band, z, tstart, twidth, scale=scale)
    recovered = recover_model(lc)
    return recovered['t_delta_rest']


def inject_model(lc, band, z, tstart, twidth, scale=1):
    """
    Injects CSM model into GALEX data and returns the resulting light curve
    Inputs:
        lc: light curve DataFrame
        band: GALEX filter
        tstart: days after discovery that ejecta impacts CSM 
        width: length of light curve plateau in days
        scale: luminosity scale factor
    """

    model = CSMmodel(tstart, twidth, DECAY_RATE, scale=scale)
    # Calculate luminosity at observation epochs
    injection = model(lc['t_delta_rest'], z)[band]
    # Inject CSM curve
    lc['luminosity_injected'] = lc['luminosity_hostsub'] + injection
    return lc


def recover_model(lc):
    """
    Recovers detections from CSM-injected data which otherwise wouldn't have
    been detected; recovered data must also be after discovery
    """

    lc['sigma_injected'] = lc['luminosity_injected'] / lc['luminosity_hostsub_err']
    recovered = lc[(lc['sigma_injected'] > SIGMA) & (lc['sigma'] < SIGMA)]
    recovered = recovered[recovered['t_delta_rest'] > 0]
    return recovered


if __name__ == '__main__':
    main()